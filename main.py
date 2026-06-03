import argparse
import torch
import numpy as np
import pickle
from util.loader import DataLoader
from util.utils import get_model_eval_metrics
import copy


def main():
    def objective(config):
        from model.ACARec import ACARec

        inp_args = copy.deepcopy(args)
        for k, v in config.items():
            vars(inp_args)[k] = v

        model = ACARec(
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
            item_artist_map,
            device,
            user_content=None,
            item_content=item_content,
        )
        model.train()
        metrics_dict = get_model_eval_metrics(model)
        return metrics_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yambda", choices=["yambda", "m4a_onion"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--topN", default="20")
    parser.add_argument("--bs", type=int, default=1024, help="training batch size")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--use_gpu", default=True, help="Whether to use CUDA")
    parser.add_argument("--save_emb", default=False)
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA id")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument("--runs_per_gpu", type=int, default=2)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument(
        "--backbone", default="BPR", type=str, help="pt file containing CF embs"
    )

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    print(args)

    device = torch.device(
        "cuda:%d" % (args.gpu_id)
        if (torch.cuda.is_available() and args.use_gpu)
        else "cpu"
    )
    # data loader
    training_data = DataLoader.load_data_set(f"./data/{args.dataset}/warm_train.csv")

    all_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/overall_val.csv")
    warm_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/warm_val.csv")
    cold_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_val.csv")
    all_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/overall_test.csv")
    warm_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/warm_test.csv")
    cold_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/cold_test.csv")

    # dataset information
    data_info_dict = pickle.load(open(f"./data/{args.dataset}/info_dict.pkl", "rb"))
    user_num = data_info_dict["user_num"]
    item_num = data_info_dict["item_num"]
    warm_user_idx = data_info_dict["warm_user"]
    warm_item_idx = data_info_dict["warm_item"]
    cold_user_idx = data_info_dict["cold_user"]
    cold_item_idx = data_info_dict["cold_item"]
    print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

    item_artist_map = np.load(f"./data/{args.dataset}/item_artist_mapping.npy")
    item_content = np.load(f"./data/{args.dataset}/embs.npy").astype(np.float32)

    config = {
        "n_artist_items": 5,
        "num_heads": 8,
        "emb_size": 128 if "yambda" in args.dataset else 512,
        "val_artist_tracks": 250,
        "use_self_attn": True,
        "backbone": args.backbone,
        "residual": False,
        "gru_mix": True,
        "glu_mix": False,
        "content_input": True,
        "bs": 1024,
    }

    metrics_dict = objective(config)
    print(metrics_dict)


if __name__ == "__main__":
    main()
