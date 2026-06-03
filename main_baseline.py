import argparse
import torch
import numpy as np
import pickle
from util.loader import DataLoader
from util.utils import get_model_eval_metrics
import copy
from functools import partial


def main():
    def objective(config):
        from model.Heater import Heater
        from model.GAR import GAR
        from model.DeepMusic import DeepMusic
        from model.CLCRec import CLCRec

        inp_args = copy.deepcopy(args)
        for k, v in config.items():
            vars(inp_args)[k] = v

        model = eval(inp_args.model)(
            inp_args,
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
            item_content=item_content,
        )
        model.train()
        out_dict = get_model_eval_metrics(model)

        return out_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--topN", default="20")
    parser.add_argument("--bs", type=int, default=2048, help="training batch size")
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--runs", type=int, default=1, help="model runs")
    parser.add_argument("--save_emb", default=False)
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA id")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--feat_dir", type=str, default="feats", help="Feat file location"
    )
    parser.add_argument(
        "--cold_object", default="item", type=str, choices=["user", "item"]
    )
    parser.add_argument("--model", default="DeepMusic", type=str)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--cf_embs_file", type=str, help="pt file containing CF embs")
    parser.add_argument(
        "--artist_mean_embs_file", type=str, help="pt file containing artist means"
    )
    parser.add_argument("--use_artist_mean", type=bool)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    print(args)

    artist_mean = str(args.use_artist_mean)
    dataset = args.dataset

    device = torch.device(
        "cuda:%d" % (args.gpu_id)
        if (torch.cuda.is_available() and args.use_gpu)
        else "cpu"
    )
    # data loader
    training_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv"
    )
    all_valid_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv"
    )
    warm_valid_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv"
    )
    cold_valid_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv"
    )
    all_test_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv"
    )
    warm_test_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv"
    )
    cold_test_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv"
    )

    # dataset information
    data_info_dict = pickle.load(
        open(f"./data/{args.dataset}/cold_{args.cold_object}/info_dict.pkl", "rb")
    )
    user_num = data_info_dict["user_num"]
    item_num = data_info_dict["item_num"]
    warm_user_idx = data_info_dict["warm_user"]
    warm_item_idx = data_info_dict["warm_item"]
    cold_user_idx = data_info_dict["cold_user"]
    cold_item_idx = data_info_dict["cold_item"]
    print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

    item_content = np.load("./data/%s/muq_embs.npy" % args.dataset).astype(np.float32)
    if args.model == "CLCRec":
        configs = {
            "m4a_onion": {
                "temp_value": 1.0,
                "lr_lambda": 0.2,
                "num_sample": 0.9,
                "lr": 0.001,
                "reg": 0.0001,
            },
            "yambda": {
                "temp_value": 1.0,
                "lr_lambda": 0.6,
                "num_sample": 0.6,
                "lr": 0.001,
                "reg": 0.0001,
            },
        }

        args.patience = 5
        config = {
            "num_neg": 128,
            "emb_size": 128 if args.dataset == "yambda" else 512,
        }
        config = {**config, **configs[dataset]}

    elif args.model == "Heater":
        configs = {
            "m4a_onion": {
                "False": {
                    "alpha": 0.001,
                    "n_dropout": 0.5,
                    "n_expert": 7,
                    "lr": 0.0001,
                    "reg": 0.00001,
                    "use_artist_mean": False,
                },
                "True": {
                    "alpha": 0.0001,
                    "n_dropout": 0.5,
                    "n_expert": 5,
                    "lr": 0.0001,
                    "reg": 0.00001,
                    "use_artist_mean": True,
                },
            },
            "yambda": {
                "False": {
                    "alpha": 0.00001,
                    "n_dropout": 0.8,
                    "n_expert": 7,
                    "use_artist_mean": False,
                    "lr": 0.001,
                    "reg": 0.0001,
                },
                "True": {
                    "alpha": 0.001,
                    "n_dropout": 0.8,
                    "n_expert": 7,
                    "use_artist_mean": True,
                    "lr": 0.001,
                    "reg": 0.0001,
                },
            },
        }
        args.bs = 1024
        config = {
            "emb_size": 128 if args.dataset == "yambda" else 512,
        }
        config = {**config, **configs[dataset][artist_mean]}
    elif args.model == "GAR":
        configs = {
            "m4a_onion": {
                "False": {
                    "alpha": 1.0,
                    "beta": 0.0,
                    "lr": 0.0001,
                    "reg": 0.000001,
                    "use_artist_mean": False,
                },
                "True": {
                    "alpha": 1.0,
                    "beta": 0.6,
                    "lr": 0.0001,
                    "reg": 0.00001,
                    "use_artist_mean": True,
                },
            },
            "yambda": {
                "False": {
                    "alpha": 1.0,
                    "beta": 0.6,
                    "use_artist_mean": False,
                    "lr": 0.001,
                    "reg": 0.0001,
                },
                "True": {
                    "alpha": 1.0,
                    "beta": 0.8,
                    "use_artist_mean": True,
                    "lr": 0.0001,
                    "reg": 0.00001,
                },
            },
        }
        args.bs = 1024
        args.patience = 10
        config = {
            "emb_size": 128 if args.dataset == "yambda" else 512,
        }

        config = {**config, **configs[dataset][artist_mean]}
    elif args.model == "DeepMusic":
        configs = {
            "m4a_onion": {
                "False": {"use_artist_mean": False, "lr": 0.001, "reg": 0.0},
                "True": {"use_artist_mean": True, "lr": 0.001, "reg": 0.000001},
            },
            "yambda": {
                "False": {"use_artist_mean": False, "lr": 0.0003, "reg": 0.0001},
                "True": {"use_artist_mean": True, "lr": 0.0003, "reg": 0.000001},
            },
        }

        config = {
            "emb_size": 128 if args.dataset == "yambda" else 512,
        }
        config = {**config, **configs[dataset][artist_mean]}

    metrics_dict = objective(config)
    print(metrics_dict)


if __name__ == "__main__":
    main()
