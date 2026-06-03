import torch
import torch.nn as nn
from .BaseRecommenderAE import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, mse_loss
import ray
import pickle
import numpy as np
import torch.nn.functional as F


def build_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    act="tanh",
    drop_rate=0.1,
    bn_first=True,
    bn_second=False,
):
    layers = []
    if bn_first:
        layers.append(nn.BatchNorm1d(input_dim))
    layers.append(nn.Linear(input_dim, hidden_dim))
    if act == "relu":
        layers.append(nn.LeakyReLU(0.01))
    if bn_second:
        layers.append(nn.BatchNorm1d(hidden_dim))
    if act == "tanh":
        layers.append(nn.Tanh())
    layers.append(nn.Dropout(drop_rate))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# Following the source code process: https://github.com/zfnWong/GAR
class GAR(BaseColdStartTrainer):
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
        device,
        user_content=None,
        item_content=None,
    ):
        super(GAR, self).__init__(
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
            device,
            user_content=user_content,
            item_content=item_content,
        )

        self.model = GAR_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        g_optimizer = torch.optim.Adam(
            model.generator.parameters(), lr=self.lr, weight_decay=self.args.reg
        )
        d_optimizer = torch.optim.Adam(
            model.mlps.parameters(), lr=self.lr, weight_decay=self.args.reg
        )
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                # Discriminator
                user_emb_r, pos_item_emb_r, neg_item_emb_r, pos_item_emb = (
                    model.get_training_embs_r(user_idx, pos_idx, neg_idx)
                )
                with torch.no_grad():
                    gen_emb = model.generator(model.item_content[pos_idx])
                gen_emb_r = model.mlps["item"](gen_emb)
                rec_loss_adv = bpr_loss(user_emb_r, pos_item_emb_r, gen_emb_r)
                rec_loss_r = bpr_loss(user_emb_r, pos_item_emb_r, neg_item_emb_r)
                rec_loss = (
                    1 - self.args.beta
                ) * rec_loss_adv + self.args.beta * rec_loss_r

                # Generator
                gen_emb = model.generator(model.item_content[pos_idx])
                for param in model.mlps["item"].parameters():
                    param.requires_grad = False
                gen_emb_r = model.mlps["item"](gen_emb)

                with torch.no_grad():
                    user_emb_r, pos_item_emb_r, neg_item_emb_r, pos_item_emb = (
                        model.get_training_embs_r(user_idx, pos_idx, neg_idx)
                    )

                gen_loss_adv = bpr_loss(user_emb_r, gen_emb_r, pos_item_emb_r)
                gen_loss_sim = mse_loss(pos_item_emb, gen_emb)
                gen_loss = (
                    1 - self.args.alpha
                ) * gen_loss_adv + self.args.alpha * gen_loss_sim

                d_optimizer.zero_grad()
                rec_loss.backward()
                d_optimizer.step()

                g_optimizer.zero_grad()
                gen_loss.backward()
                g_optimizer.step()

            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb = self.model()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                cold_item_gen_emb = model.generate_item_emb(
                    self.data.mapped_cold_item_idx
                )
                self.item_emb.data[self.data.mapped_cold_item_idx] = cold_item_gen_emb
                if epoch % self.args.eval_freq == 0:
                    self.fast_evaluation_quiet(epoch, valid_type="cold")
            if epoch + 1 - self.bestPerformance[0] >= self.args.patience:
                break

        self.timer(start=False)
        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            now_best_user_emb, now_best_item_emb = self.model.forward()
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()
            if self.args.cold_object == "item":
                now_cold_item_gen_emb = self.model.generate_item_emb(
                    self.data.mapped_cold_item_idx
                )
                self.best_item_emb.data[self.data.mapped_cold_item_idx] = (
                    now_cold_item_gen_emb
                )
            else:
                now_cold_user_gen_emb = self.model.generate_user_emb(
                    self.data.mapped_cold_user_idx
                )
                self.best_user_emb.data[self.data.mapped_cold_user_idx] = (
                    now_cold_user_gen_emb
                )

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class GAR_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(GAR_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        self.cf_embs_file = args.cf_embs_file
        if self.args.cold_object == "item":
            self.item_content = torch.tensor(
                self.data.mapped_item_content, dtype=torch.float32, requires_grad=False
            ).to(device)
        else:
            self.user_content = torch.tensor(
                self.data.mapped_user_content, dtype=torch.float32, requires_grad=False
            ).to(device)

        if args.use_artist_mean:
            artist_means = torch.load(args.artist_mean_embs_file).cuda()
            self.item_content = torch.cat([artist_means, self.item_content], dim=1)
        self.content_dim = self.item_content.shape[1]
        self.generator = build_mlp(
            self.content_dim,
            self.content_dim,
            self.latent_size,
            drop_rate=0.1,
            bn_first=False,
            bn_second=True,
        )

        item_mlp = nn.Linear(self.latent_size, self.latent_size)
        user_mlp = nn.Linear(self.latent_size, self.latent_size)
        with torch.no_grad():
            item_mlp.weight.copy_(torch.eye(self.latent_size))
            item_mlp.bias.copy_(torch.zeros_like(item_mlp.bias))
            user_mlp.weight.copy_(torch.eye(self.latent_size))
            user_mlp.bias.copy_(torch.zeros_like(user_mlp.bias))
        self.mlps = nn.ModuleDict({"item": item_mlp, "user": user_mlp})
        self.embedding_dict = self._init_model()

    def _init_model(self):
        user_emb, item_emb = torch.load(self.cf_embs_file, map_location="cpu")
        item_emb_mapped = torch.zeros(*item_emb.shape)
        for i, ind in enumerate([int(k) for k in self.data.item]):
            item_emb_mapped[i] = item_emb[ind]
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": user_emb[[int(k) for k in self.data.user]],
                "item_emb": item_emb_mapped,
            }
        )
        embedding_dict["user_emb"].requires_grad = False
        embedding_dict["item_emb"].requires_grad = False
        return embedding_dict

    def get_training_embs_r(self, uid, iid, nid):
        user_emb_r = self.mlps["user"](self.embedding_dict["user_emb"][uid])
        pos_item_emb = self.embedding_dict["item_emb"][iid]
        pos_item_emb_r = self.mlps["item"](pos_item_emb)
        neg_item_emb_r = self.mlps["item"](self.embedding_dict["item_emb"][nid])
        return user_emb_r, pos_item_emb_r, neg_item_emb_r, pos_item_emb

    def forward(self):
        return self.mlps["user"](self.embedding_dict["user_emb"]), self.mlps["item"](
            self.embedding_dict["item_emb"]
        )

    def generate_item_emb(self, gen_idx):
        return self.mlps["item"](self.generator(self.item_content[gen_idx]))

    def generate_user_emb(self, gen_idx):
        return self.generator(self.user_content[gen_idx])
