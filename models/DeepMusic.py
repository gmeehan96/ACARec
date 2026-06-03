import torch
import torch.nn as nn
from .BaseRecommenderAE import BaseColdStartTrainer
from util.utils import next_batch_item_only
import numpy as np
import torch.nn.functional as F


class DeepMusic(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(DeepMusic, self).__init__(
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

        self.model = DeepMusic_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.args.reg)
        self.timer(start=True)
        loss_fn = F.mse_loss
        self.user_emb, self.item_emb = self.model.embedding_dict['user_emb'],self.model.embedding_dict['item_emb']
        for epoch in range(self.maxEpoch):
            losses = []
            model.train()

            for n, item_idx in enumerate(next_batch_item_only(self.data, self.batch_size)):
                e_hat = model(item_idx)
                e_target = self.model.embedding_dict['item_emb'][item_idx]
                loss = loss_fn(e_hat,e_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            with torch.no_grad():
                model.eval()
                cold_outputs = []
                for n, item_idx in enumerate(next_batch_item_only(self.data, self.batch_size,training=False)):
                    e_hat = model(item_idx)
                    cold_outputs.append(e_hat)
                cold_emb = torch.cat(cold_outputs)
                self.item_emb.data[self.data.mapped_cold_item_idx] = cold_emb

                if epoch % self.args.eval_freq == 0:
                    measure,performance = self.fast_evaluation_quiet(epoch, valid_type='cold')

            if epoch + 1 - self.bestPerformance[0] >= self.args.patience:
                break
        
        model.eval()
        self.item_emb = self.best_item_emb
        self.timer(start=False)

    def save(self):
        self.best_item_emb = self.item_emb.clone()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class DeepMusic_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(DeepMusic_Learner, self).__init__()
        self.args = args
        self.hidden_dim = emb_size
        self.device = device
        self.data = data
        self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        self.cf_embs_file = args.cf_embs_file

        if args.use_artist_mean:
            self.artist_means = torch.load(args.artist_mean_embs_file).cuda()
            self.item_content = torch.cat([self.artist_means,self.item_content],dim=1)
        self.content_dim = self.item_content.shape[1]
        
        self.embedding_dict = self._init_model()
        self.collab_dim = self.embedding_dict['item_emb'].shape[1]

        self.content_trans = nn.Sequential(nn.Linear(self.content_dim,self.content_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.content_dim, self.collab_dim))


    def _init_model(self):
        user_emb,item_emb = torch.load(self.cf_embs_file, map_location='cpu')
        if 'BPR' in self.args.backbone:
            item_emb_mapped = torch.zeros(*item_emb.shape)
            for i,ind in enumerate([int(k) for k in self.data.item]):
                item_emb_mapped[i] = item_emb[ind]
            embedding_dict = nn.ParameterDict({
                'user_emb': user_emb[[int(k) for k in self.data.user]],
                'item_emb': self.args.emb_scale*item_emb_mapped,
            })
        else:
            embedding_dict = nn.ParameterDict({
                'user_emb': user_emb,
                'item_emb': self.args.emb_scale*item_emb,
            })
        embedding_dict['user_emb'].requires_grad = False
        embedding_dict['item_emb'].requires_grad = False
        return embedding_dict


    def forward(self,item_idx_batch):
        out = self.content_trans(self.item_content[item_idx_batch])
        return out

