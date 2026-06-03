import torch
import torch.nn as nn
from .BaseRecommenderAE import BaseColdStartTrainer_Artist
from util.utils import next_batch_artist
import torch.nn.functional as F
import copy


class ACARec(BaseColdStartTrainer_Artist):
    def __init__(
        self,
        args,
        training_data,
        warm_valid_data,
        cold_valid_data,
        all_valid_data,
        warm_test_data,
        cold_test_data,
        all_test_data,
        user_num,
        item_num,
        warm_user_idx,
        warm_item_idx,
        cold_user_idx,
        cold_item_idx,
        item_artist_map,
        device,
        user_content=None,
        item_content=None,
    ):
        super(ACARec, self).__init__(
            args,
            training_data,
            warm_valid_data,
            cold_valid_data,
            all_valid_data,
            warm_test_data,
            cold_test_data,
            all_test_data,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            item_artist_map,
            device,
            user_content=user_content,
            item_content=item_content,
        )

        self.model = ACARec_Learner(
            args, self.data, self.emb_size, args.num_heads, args.use_self_attn, device
        )
        self.n_artist_items = args.n_artist_items
        self.val_artist_tracks = args.val_artist_tracks

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.args.reg
        )
        loss_fn = F.mse_loss
        self.user_emb, self.item_emb = (
            self.model.embedding_dict["user_emb"],
            self.model.embedding_dict["item_emb"],
        )
        for epoch in range(self.maxEpoch):
            losses = []
            model.train()

            for n, (item_idx, artist_item_idx) in enumerate(
                next_batch_artist(
                    self.data,
                    self.batch_size,
                    self.n_artist_items,
                    n_negs=self.args.n_negs,
                )
            ):
                e_hat, attn_weights, entropy = model(item_idx, artist_item_idx)

                e_target = self.model.embedding_dict["item_emb"][item_idx]
                loss = loss_fn(e_hat, e_target)
                total_loss = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            with torch.no_grad():
                model.eval()
                cold_outputs = []
                for n, (item_idx, artist_item_idx) in enumerate(
                    next_batch_artist(
                        self.data,
                        self.batch_size,
                        self.n_artist_items,
                        training=False,
                        val_artist_tracks=self.val_artist_tracks,
                    )
                ):
                    e_hat, attn_weights, _ = model(item_idx, artist_item_idx)
                    cold_outputs.append(e_hat)
                cold_emb = torch.cat(cold_outputs)
                self.item_emb.data[self.data.mapped_cold_item_idx] = cold_emb

                if epoch % self.args.eval_freq == 0:
                    measure, performance = self.fast_evaluation_quiet(
                        epoch, valid_type="cold"
                    )

            if epoch + 1 - self.bestPerformance[0] >= self.args.patience:
                break

        model.eval()
        self.item_emb = self.best_item_emb

    def save(self):
        self.best_item_emb = self.item_emb.clone()
        self.best_state_dict = copy.deepcopy(self.model.state_dict())

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class ACARec_Learner(nn.Module):
    def __init__(self, args, data, emb_size, num_heads, device):
        super(ACARec_Learner, self).__init__()
        self.args = args
        self.hidden_dim = emb_size
        self.device = device
        self.data = data
        self.item_content = torch.tensor(
            self.data.mapped_item_content, dtype=torch.float32, requires_grad=False
        ).to(device)
        self.content_dim = self.item_content.shape[1]
        self.residual = args.residual

        self.embedding_dict = self._init_model()
        self.collab_dim = self.embedding_dict["item_emb"].shape[1]
        self.use_self_attn = args.use_self_attn

        self.key_proj = nn.Linear(self.content_dim + self.collab_dim, self.hidden_dim)
        self.val_proj = nn.Linear(self.collab_dim, self.hidden_dim)
        self.query_proj = nn.Linear(self.content_dim, self.collab_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.collab_dim)

        if self.use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                self.hidden_dim,
                num_heads,
                batch_first=True,
            )

        if self.args.gru_mix:
            if self.args.content_input:
                self.gru = nn.GRUCell(
                    self.content_dim + self.collab_dim, self.collab_dim
                )
            else:
                self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        elif self.args.glu_mix:
            self.glu = GLUFusion(
                self.content_dim + 2 * self.collab_dim, self.collab_dim
            )

        self.cross_attn = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads,
            batch_first=True,
            kdim=self.hidden_dim,
            vdim=self.collab_dim,
        )

        if self.args.content_input:
            self.content_merge = nn.Linear(
                self.content_dim + self.collab_dim, self.collab_dim
            )

    def _init_model(self):
        user_emb, item_emb = torch.load(
            f"./data/{self.args.dataset}/embs/{self.args.backbone}.pt",
            map_location="cpu",
        )

        embedding_dict = nn.ParameterDict(
            {
                "user_emb": user_emb,
                "item_emb": item_emb,
            }
        )
        return embedding_dict

    def build_artist_inputs(
        self,
        item_idx_batch,
        artist_item_idx_batch,
        pad_value: int = -1,
    ):
        """
        Args:
            item_idx_batch:            (B,) list or LongTensor
            artist_item_idx_batch:     (B, K) LongTensor (padded with pad_value)
            pad_value:                 int (default: -1)

        Returns:
            c_t  : (B, D_c)
            c_i  : (B, K, D_c)
            e_i  : (B, K, D_e)
            mask : (B, K) bool
        """

        device = self.item_content.device

        item_idx_batch = torch.as_tensor(
            item_idx_batch, dtype=torch.long, device=device
        )
        c_t = self.item_content[item_idx_batch]  # (B, D_c)

        artist_item_idx_batch = torch.as_tensor(
            artist_item_idx_batch, dtype=torch.long, device=device
        )

        # Mask: True for real tracks
        mask = artist_item_idx_batch != pad_value  # (B, K)
        safe_idx = artist_item_idx_batch.clamp(min=0)

        # ---- Embedding lookup ----
        c_i = self.item_content[safe_idx]  # (B, K, D_c)
        e_i = self.embedding_dict["item_emb"][safe_idx]  # (B, K, D_e)

        # Zero out padded positions
        c_i = c_i * mask.unsqueeze(-1)
        e_i = e_i * mask.unsqueeze(-1)

        return c_t, c_i, e_i, mask

    def forward(self, item_idx_batch, artist_item_idx_batch):
        c_t, c_i, e_i, mask = self.build_artist_inputs(
            item_idx_batch, artist_item_idx_batch
        )
        B, K, _ = c_i.shape

        if self.residual or self.args.gru_mix or self.args.glu_mix:
            mask_f = mask.float()
            denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            artist_mean = (e_i * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B, D_e)

        # ---- Build keys / values ----
        keys = self.key_proj(torch.cat([c_i, e_i], dim=-1))  # (B, K, H)
        values = self.val_proj(e_i)  # (B, K, H)

        # ---- Self-attention over artist set ----
        if self.use_self_attn:
            keys, _ = self.self_attn(
                keys,
                keys,
                keys,
                key_padding_mask=~mask,
                need_weights=False,
            )

        # ---- Query from cold track content ----
        query = self.query_proj(c_t).unsqueeze(1)  # (B, 1, H)

        # ---- Cross-attention ----
        attn_out, _ = self.cross_attn(
            query,
            keys,
            values,
            key_padding_mask=~mask,
            need_weights=False,
        )

        attn_out = attn_out.squeeze(1)  # (B, H)
        e_hat = self.out_proj(attn_out)  # (B, D_e)

        if self.args.content_input:
            final_input = torch.cat([e_hat, c_t], dim=1)
            if not self.args.gru_mix:
                e_hat = self.content_merge(final_input)
        else:
            final_input = e_hat

        if self.residual:
            e_hat += artist_mean
        elif self.args.gru_mix:
            e_hat = self.gru(final_input, artist_mean)
        elif self.args.glu_mix:
            e_hat = self.glu(artist_mean, final_input)

        return e_hat


class GLUFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, e_artist, e_update):
        h = torch.cat([e_artist, e_update], dim=-1)
        u_pre, z_pre = self.proj(h).chunk(2, dim=-1)
        z = torch.sigmoid(z_pre)
        return (1 - z) * e_artist + z * u_pre
