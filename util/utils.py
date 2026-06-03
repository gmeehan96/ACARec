from random import shuffle, randint, choice, sample
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from functools import partial

def next_batch_artist(
    data,
    batch_size,
    n_artist_tracks=5,
    pad_value=-1,
    training=True,
    val_artist_tracks=250,
):
    if training:
        item_list = list(data.source_warm_item_idx)
        data_size = len(item_list)
        items = sample(item_list, data_size)
    else:
        items = list(data.source_cold_item_idx)
        data_size = len(items)
        n_artist_tracks = val_artist_tracks

    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size

        i_idx_unmapped = [items[i] for i in range(ptr, batch_end)]
        ptr = batch_end
        i_idx = []
        artist_i_idx = []
        for item in i_idx_unmapped:
            item_artist_tracks = data.artist_track_map[data.track_artist_map[item]]
            item_artist_tracks = item_artist_tracks.difference([item])
            if len(item_artist_tracks) > 0:
                num_sample = min(len(list(item_artist_tracks)), n_artist_tracks)
                artist_i_idx_sampled = sample(item_artist_tracks, num_sample)
                artist_i_idx_sampled = [data.item[i] for i in artist_i_idx_sampled]
                while len(artist_i_idx_sampled) < n_artist_tracks:
                    artist_i_idx_sampled.append(-1)

                i_idx.append(data.item[item])
                artist_i_idx.append(artist_i_idx_sampled)

        yield i_idx, np.array(artist_i_idx, dtype=np.int32)


def set_seed(seed, cuda):
    print("Set Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    zeros = torch.zeros(sparse_mx.shape, dtype=torch.float32)
    values = torch.from_numpy(sparse_mx.data)
    zeros[sparse_mx.row, sparse_mx.col] = values
    return zeros


def get_model_eval_metrics(model):
    num_batches = 100
    batch_size = 1 + model.data.interaction_mat.shape[0] // num_batches
    out_dict = {}
    eval_fns = {
        "val": partial(
            model.eval_valid_new,
            **{"batch_size": batch_size, "topk_warm": 50, "topk_cold": 20}
        ),
        "test": partial(
            model.eval_test_new,
            **{"batch_size": batch_size, "topk_warm": 50, "topk_cold": 20}
        ),
    }

    for split in ["val", "test"]:
        eval_fns[split]()
        if split == "val":
            warm_recall, warm_ndcg = model.warm_valid_results
            cold_recall, cold_ndcg = model.cold_valid_results
        if split == "test":
            warm_recall, warm_ndcg = model.warm_test_results
            cold_recall, cold_ndcg = model.cold_test_results
        out_dict = {
            **out_dict,
            **{
                "%s_warm_recall" % split: warm_recall,
                "%s_warm_ndcg" % split: warm_ndcg,
                "%s_cold_recall" % split: cold_recall,
                "%s_cold_ndcg" % split: cold_ndcg,
            },
        }
    return out_dict

def next_batch_item_only(data,batch_size,training=True):
    if training:
        item_list = list(data.source_warm_item_idx)
        data_size = len(item_list)
        items = sample(item_list,data_size)
    else:
        items = list(data.source_cold_item_idx)
        data_size = len(items)
    
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        
        i_idx = [data.item[items[i]] for i in range(ptr, batch_end)]
        ptr = batch_end
        yield i_idx


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def mse_loss(real_item_emb, item_content_emb):
    loss = F.mse_loss(real_item_emb, item_content_emb)
    return loss

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []

        item_list = list(data.source_warm_item_idx)
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_CLCRec(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(data.source_warm_item_idx)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx = [], []
        for i, user in enumerate(users):
            u_idx.append([data.user[user]]*(1+n_negs))
            i_idx.append([data.item[items[i]]])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                i_idx[i].append(data.item[neg_item])
        yield u_idx, i_idx
