import sys

sys.path.append("..")
from util.databuilder import ColdStartDataBuilder
from util.artist_databuilder import ColdStartDataBuilder_Artist
from util.evaluator import ranking_evaluation
from util.utils import sparse_mx_to_torch_sparse_tensor
import torch
import time


class BaseColdStartTrainer(object):
    def __init__(
        self,
        args,
        training_set,
        warm_valid_set,
        cold_valid_set,
        overall_valid_set,
        warm_test_set,
        cold_test_set,
        overall_test_set,
        user_num,
        item_num,
        warm_user_idx,
        warm_item_idx,
        cold_user_idx,
        cold_item_idx,
        device,
        user_content=None,
        item_content=None,
        **kwargs
    ):
        super(BaseColdStartTrainer, self).__init__()
        self.args = args
        self.data = ColdStartDataBuilder(
            training_set,
            warm_valid_set,
            cold_valid_set,
            overall_valid_set,
            warm_test_set,
            cold_test_set,
            overall_test_set,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            user_content,
            item_content,
        )
        self.bestPerformance = []
        top = args.topN.split(",")
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.emb_size = args.emb_size
        self.maxEpoch = args.epochs
        self.batch_size = args.bs
        self.lr = args.lr
        self.reg = args.reg
        self.device = device
        self.result = []

    def print_basic_info(self):
        print("*" * 80)
        print("Model: ", self.model_name)
        print("Dataset: ", self.dataset_name)
        print("Embedding Dimension:", self.emb_size)
        print("Maximum Epoch:", self.maxEpoch)
        print("Learning Rate:", self.lr)
        print("Batch Size:", self.batch_size)
        print("*" * 80)

    def timer(self, start=True):
        if start:
            self.train_start_time = time.time()
        else:
            self.train_end_time = time.time()

    def train(self):
        pass

    def predict(self, u):
        pass

    def save(self):
        pass

    def fast_evaluation_quiet(self, epoch, valid_type="all"):
        if valid_type == "warm":
            valid_set = self.data.warm_valid_set
        elif valid_type == "cold":
            valid_set = self.data.cold_valid_set
        elif valid_type == "all":
            valid_set = self.data.overall_valid_set
        else:
            raise ValueError("Invalid evaluation type!")
        rec_list = self.valid(valid_type)
        measure, _ = ranking_evaluation(valid_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(":")
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(":")
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        measure = [m.strip() for m in measure[1:]]
        return measure, performance

    @torch.no_grad()
    def eval_valid(self, batch_size=5000, topk_warm=20, topk_cold=20):
        warm_item_emb = self.item_emb[self.data.mapped_warm_item_idx]
        cold_item_emb = self.item_emb[self.data.mapped_cold_valid_item_idx]

        num_batches = 1 + self.data.user_num // batch_size
        warm_out = []
        cold_out = []
        for b in range(num_batches):
            user_emb_batch = self.user_emb[b * batch_size : (b + 1) * batch_size]
            warm_scores = user_emb_batch @ warm_item_emb.T
            mask = sparse_mx_to_torch_sparse_tensor(
                self.data.interaction_mat[b * batch_size : (b + 1) * batch_size][
                    :, self.data.mapped_warm_item_idx
                ]
            ).cuda()
            warm_scores += -1e10 * mask
            warm_out.append(torch.topk(warm_scores, topk_warm))
            cold_scores = user_emb_batch @ cold_item_emb.T
            cold_out.append(torch.topk(cold_scores, topk_cold))

        warm_vals, warm_inds = [torch.cat(x) for x in list(zip(*warm_out))]
        warm_rec_list = self.get_rec_list_full(
            self.data.warm_valid_set,
            warm_inds,
            warm_vals,
            self.data.mapped_warm_item_idx,
        )
        self.warm_valid_results = ranking_evaluation(
            self.data.warm_valid_set, warm_rec_list, [topk_warm]
        )[1][0]

        cold_vals, cold_inds = [torch.cat(x) for x in list(zip(*cold_out))]
        cold_rec_list = self.get_rec_list_full(
            self.data.cold_valid_set,
            cold_inds,
            cold_vals,
            self.data.mapped_cold_valid_item_idx,
        )
        self.cold_valid_results = ranking_evaluation(
            self.data.cold_valid_set, cold_rec_list, [topk_cold]
        )[1][0]

    @torch.no_grad()
    def eval_test(self, batch_size=5000, topk_warm=20, topk_cold=20):
        warm_item_emb = self.item_emb[self.data.mapped_warm_item_idx]
        cold_item_emb = self.item_emb[self.data.mapped_cold_test_item_idx]

        num_batches = 1 + self.data.user_num // batch_size
        warm_out = []
        cold_out = []
        for b in range(num_batches):
            user_emb_batch = self.user_emb[b * batch_size : (b + 1) * batch_size]
            warm_scores = user_emb_batch @ warm_item_emb.T
            mask = sparse_mx_to_torch_sparse_tensor(
                self.data.interaction_mat[b * batch_size : (b + 1) * batch_size][
                    :, self.data.mapped_warm_item_idx
                ]
            ).cuda()
            warm_scores += -1e10 * mask
            warm_out.append(torch.topk(warm_scores, topk_warm))
            cold_scores = user_emb_batch @ cold_item_emb.T
            cold_out.append(torch.topk(cold_scores, topk_cold))

        warm_vals, warm_inds = [torch.cat(x) for x in list(zip(*warm_out))]
        warm_rec_list = self.get_rec_list_full(
            self.data.warm_test_set,
            warm_inds,
            warm_vals,
            self.data.mapped_warm_item_idx,
        )
        self.warm_test_results = ranking_evaluation(
            self.data.warm_test_set, warm_rec_list, [topk_warm]
        )[1][0]

        cold_vals, cold_inds = [torch.cat(x) for x in list(zip(*cold_out))]
        cold_rec_list = self.get_rec_list_full(
            self.data.cold_test_set,
            cold_inds,
            cold_vals,
            self.data.mapped_cold_test_item_idx,
        )
        self.cold_test_results = ranking_evaluation(
            self.data.cold_test_set, cold_rec_list, [topk_cold]
        )[1][0]

    def get_rec_list(self, u, top_inds, top_values_final, item_inds):
        u_mapped = self.data.get_user_id(u)
        items = top_inds[u_mapped]
        items_mapped = [self.data.id2item[item_inds[iid]] for iid in items]
        return list(zip(items_mapped, top_values_final[u_mapped]))

    def get_rec_list_full(self, test_set, top_inds, top_values_final, item_inds):
        rec_list = {}
        for i, user in enumerate(test_set):
            rec_list[user] = self.get_rec_list(
                user, top_inds, top_values_final, item_inds
            )
        return rec_list


class BaseColdStartTrainer_Artist(BaseColdStartTrainer):
    def __init__(
        self,
        args,
        training_set,
        warm_valid_set,
        cold_valid_set,
        overall_valid_set,
        warm_test_set,
        cold_test_set,
        overall_test_set,
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
        **kwargs
    ):

        super(BaseColdStartTrainer_Artist, self).__init__(
            args,
            training_set,
            warm_valid_set,
            cold_valid_set,
            overall_valid_set,
            warm_test_set,
            cold_test_set,
            overall_test_set,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            device,
            user_content,
            item_content,
            **kwargs
        )

        self.data = ColdStartDataBuilder_Artist(
            training_set,
            warm_valid_set,
            cold_valid_set,
            overall_valid_set,
            warm_test_set,
            cold_test_set,
            overall_test_set,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            item_artist_map,
            user_content,
            item_content,
        )
