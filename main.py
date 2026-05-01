import os
import json
import argparse
import numpy as np
import torch

from utility.data_loader import load_dataset, load_subject_groups
from utility.trial_selector import load_trial_info, add_non_bio_ranks
from decoder.group_decoder import run_group_decoding_cv


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]

    return obj


def summarize_fold_results(fold_results):
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "auc_roc",
        "f1",
        "precision",
        "recall",
    ]

    summary = {}

    for metric in metric_names:
        vals = [
            fold["metrics"][metric]
            for fold in fold_results
            if fold["metrics"][metric] is not None
        ]

        if len(vals) == 0:
            summary[metric] = {
                "mean": None,
                "std": None,
            }
        else:
            summary[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    return summary


def main(args):

    os.makedirs(args.save_path, exist_ok=True)

    Dataset = load_dataset(
        fPath=args.fPath,
        fileName=args.fileName,
        bPath=args.bPath,
        chName=args.chName,
        t_end_ms=args.t_end_ms,
    )

    group_indices = load_subject_groups(
        bPath=args.bPath,
        subIdx=args.subIdx,
    )

    trialInfo = load_trial_info(args.trialInfo)

    comparisons = [
        ("Control_vs_Depressed", "Control", "Depressed"),
        ("Control_vs_Suicidal", "Control", "Suicidal"),
        ("Depressed_vs_Suicidal", "Depressed", "Suicidal"),
    ]

    comparison_names = [x[0] for x in comparisons]
    trialInfo = add_non_bio_ranks(trialInfo, comparison_names)

    print(f"Loaded Dataset: {Dataset.shape}")
    print(f"Loaded trialInfo: {args.trialInfo}")

    if args.toi_mode == "all":
        chunk_ids = [0, 1, 2, 3]
    elif args.toi_mode == "non_bio":
        chunk_ids = [0, 1, 2]
    else:
        raise ValueError("toi_mode must be 'all' or 'non_bio'.")

    all_results = {}

    for comparison, group_a, group_b in comparisons:

        for chunk_id in chunk_ids:

            key = (
                f"{comparison}_"
                f"chunk{chunk_id + 1}_"
                f"{args.feature_mode}_"
                f"{args.toi_mode}_"
                f"{args.model_type}"
            )

            print("\n" + "=" * 80)
            print(f"Running: {key}")
            print("=" * 80)

            curve_dir = os.path.join(args.save_path, "curves")

            fold_results = run_group_decoding_cv(
                Dataset=Dataset,
                trialInfo=trialInfo,
                group_indices=group_indices,
                comparison=comparison,
                group_a=group_a,
                group_b=group_b,
                chunk_id=chunk_id,
                feature_mode=args.feature_mode,
                toi_mode=args.toi_mode,
                n_aug_train=args.n_aug_train,
                n_aug_test=args.n_aug_test,
                k=args.k,
                n_splits=args.n_splits,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                dropout=args.dropout,
                seed=args.seed,
                device=args.device,
                val_size=args.val_size,
                patience=args.patience,
                min_delta=args.min_delta,
                curve_dir=curve_dir,
                verbose=args.verbose,
                model_type=args.model_type,
            )

            result_obj = {
                "key": key,
                "comparison": comparison,
                "group_a": group_a,
                "group_b": group_b,
                "chunk_id": int(chunk_id),
                "feature_mode": args.feature_mode,
                "toi_mode": args.toi_mode,
                "fold_results": fold_results,
                "summary": summarize_fold_results(fold_results),
                "model_type": args.model_type,
            }

            all_results[key] = result_obj

            out_file = os.path.join(args.save_path, f"{key}.json")

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(convert_numpy(result_obj), f, indent=4)

            print(f"Saved: {out_file}")

    summary_file = os.path.join(
        args.save_path,
        f"summary_{args.feature_mode}_{args.toi_mode}.json",
    )

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(convert_numpy(all_results), f, indent=4)

    print(f"\nSaved summary: {summary_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fPath",
        type=str,
        default="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
    )

    parser.add_argument(
        "--bPath",
        type=str,
        default="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/GroupDecoding/",
    )

    parser.add_argument(
        "--fileName",
        type=str,
        default="Data_sen_lepoch.pkl",
    )

    parser.add_argument(
        "--subIdx",
        type=str,
        default="subject_index.mat",
    )

    parser.add_argument(
        "--trialInfo",
        type=str,
        default="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/trial_sentence_rank_info.pkl",
    )

    parser.add_argument(
        "--feature_mode",
        type=str,
        default="uniform",
        choices=["uniform", "contrast", "sentence_response"],
    )

    parser.add_argument(
        "--toi_mode",
        type=str,
        default="all",
        choices=["all", "non_bio"],
    )

    parser.add_argument("--n_aug_train", type=int, default=100)
    parser.add_argument("--n_aug_test", type=int, default=50)
    parser.add_argument("--k", type=int, default=12)

    parser.add_argument(
        "--chName",
        type=str,
        default="GoodChannel.mat",
    )

    parser.add_argument(
        "--t_end_ms",
        type=int,
        default=1000,
    )

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.005)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="eegnet",
        choices=["deepconvnet", "eegnet"],
    )

    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_balanced_accuracy"],
    )

    parser.add_argument(
        "--curve_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()
    if args.chName.lower() == "none":
        args.chName = None

    main(args)
