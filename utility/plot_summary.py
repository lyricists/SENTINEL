# ============================================================
# Plot metric across chunks from GroupDecoding summary JSONs
# Select desired feature modes and TOI modes from command line
# ============================================================

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric_by_chunk(
    summary_data,
    comparison,
    feature_mode,
    toi_mode,
    metric,
    model_type,
):
    n_chunks = 4 if toi_mode == "all" else 3

    chunks = []
    means = []
    stds = []

    for chunk in range(1, n_chunks + 1):
        key = f"{comparison}_chunk{chunk}_{feature_mode}_{toi_mode}_{model_type}"

        chunks.append(chunk)

        if key not in summary_data:
            print(f"Missing key: {key}")
            means.append(np.nan)
            stds.append(np.nan)
            continue

        metric_info = summary_data[key]["summary"][metric]

        means.append(metric_info["mean"])
        stds.append(metric_info["std"])

    return np.array(chunks), np.array(means), np.array(stds)


def make_row_settings(feature_modes, toi_modes):
    row_settings = []

    label_map = {
        "uniform": "Uniform",
        "contrast": "Contrast",
        "sentence_response": "Sentence Response",
        "all": "All",
        "non_bio": "Non-Bio",
    }

    for feature_mode in feature_modes:
        for toi_mode in toi_modes:
            row_title = f"{label_map[feature_mode]} {label_map[toi_mode]}"
            row_settings.append((row_title, feature_mode, toi_mode))

    return row_settings


def main(args):

    metric_map = {
        "auc": "auc_roc",
        "auc_roc": "auc_roc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "balanced_accuracy": "balanced_accuracy",
        "bacc": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    metric = metric_map[args.metric]

    metric_label = {
        "auc_roc": "AUC-ROC",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "balanced_accuracy": "Balanced Accuracy",
        "accuracy": "Accuracy",
    }[metric]

    comparisons = [
        ("Control_vs_Depressed", "C vs D"),
        ("Depressed_vs_Suicidal", "D vs S"),
        ("Control_vs_Suicidal", "C vs S"),
    ]

    row_settings = make_row_settings(
        feature_modes=args.feature_modes,
        toi_modes=args.toi_modes,
    )

    n_rows = len(row_settings)
    n_cols = len(comparisons)

    summary_data = {}

    for row_title, feature_mode, toi_mode in row_settings:
        summary_file = f"summary_{feature_mode}_{toi_mode}.json"
        path = os.path.join(args.result_dir, summary_file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        summary_data[row_title] = load_summary(path)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 3.2 * n_rows),
        sharey=True,
        squeeze=False,
    )

    for row_idx, (row_title, feature_mode, toi_mode) in enumerate(row_settings):

        data = summary_data[row_title]

        for col_idx, (comparison, col_title) in enumerate(comparisons):

            ax = axes[row_idx, col_idx]

            chunks, mean_vals, std_vals = extract_metric_by_chunk(
                summary_data=data,
                comparison=comparison,
                feature_mode=feature_mode,
                toi_mode=toi_mode,
                metric=metric,
                model_type=args.model_type,
            )

            ax.plot(chunks, mean_vals, marker="o", linewidth=2)
            ax.errorbar(chunks, mean_vals, yerr=std_vals, fmt="none", capsize=4)

            ax.axhline(0.5, linestyle="--", linewidth=1)

            ax.set_xticks(chunks)
            ax.set_ylim(args.ymin, args.ymax)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=14)

            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n{metric_label}", fontsize=12)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Chunk number", fontsize=12)

            ax.grid(True, alpha=0.3)

    feature_text = "_".join(args.feature_modes)
    toi_text = "_".join(args.toi_modes)

    plt.suptitle(
        f"Mean {metric_label} Across Folds by Sentence-Ranking Chunk",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        out_png = os.path.join(
            args.result_dir,
            f"{metric}_{feature_text}_{toi_text}_{args.model_type}_summary.png",
        )
        out_pdf = out_png.replace(".png", ".pdf")

        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)

        print(f"Saved PNG: {out_png}")
        print(f"Saved PDF: {out_pdf}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_dir",
        type=str,
        default="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/GroupDecoding",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="auc_roc",
        choices=[
            "auc",
            "auc_roc",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "bacc",
            "accuracy",
        ],
    )

    parser.add_argument(
        "--feature_modes",
        nargs="+",
        default=["uniform"],
        choices=["uniform", "contrast", "sentence_response"],
        help="Choose one or more: uniform contrast sentence_response",
    )

    parser.add_argument(
        "--toi_modes",
        nargs="+",
        default=["all", "non_bio"],
        choices=["all", "non_bio"],
        help="Choose one or more: all non_bio",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="deepconvnet",
        choices=["deepconvnet", "eegnet"],
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
