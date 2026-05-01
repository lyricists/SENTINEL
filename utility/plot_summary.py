# ============================================================
# Plot metric across chunks from GroupDecoding summary JSONs
# ============================================================

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric_by_chunk(summary_data, comparison, feature_mode, toi_mode, metric):
    n_chunks = 4 if toi_mode == "all" else 3

    chunks = []
    means = []
    stds = []

    for chunk in range(1, n_chunks + 1):
        key = f"{comparison}_chunk{chunk}_{feature_mode}_{toi_mode}"

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

    summary_files = {
        "Uniform All": "summary_uniform_all.json",
        "Uniform Non-Bio": "summary_uniform_non_bio.json",
        "Contrast All": "summary_contrast_all.json",
        "Contrast Non-Bio": "summary_contrast_non_bio.json",
    }

    comparisons = [
        ("Control_vs_Depressed", "C vs D"),
        ("Depressed_vs_Suicidal", "D vs S"),
        ("Control_vs_Suicidal", "C vs S"),
    ]

    row_settings = [
        ("Uniform All", "uniform", "all"),
        ("Uniform Non-Bio", "uniform", "non_bio"),
        ("Contrast All", "contrast", "all"),
        ("Contrast Non-Bio", "contrast", "non_bio"),
    ]

    summary_data = {}

    for row_name, filename in summary_files.items():
        path = os.path.join(args.result_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        summary_data[row_name] = load_summary(path)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(15, 12),
        sharey=True,
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
            )

            ax.plot(chunks, mean_vals, marker="o", linewidth=2)
            ax.errorbar(chunks, mean_vals, yerr=std_vals, fmt="none", capsize=4)

            ax.axhline(0.5, linestyle="--", linewidth=1)

            ax.set_xticks(chunks)
            ax.set_ylim(0.0, 1.0)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=14)

            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n{metric_label}", fontsize=12)

            if row_idx == 3:
                ax.set_xlabel("Chunk number", fontsize=12)

            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Mean {metric_label} Across Folds by Sentence-Ranking Chunk",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # out_png = os.path.join(args.result_dir, f"{metric}_summary_4x3.png")
    # out_pdf = os.path.join(args.result_dir, f"{metric}_summary_4x3.pdf")

    # plt.savefig(out_png, dpi=300)
    # plt.savefig(out_pdf)

    if args.show:
        plt.show()
    else:
        plt.close()

    # print(f"Saved PNG: {out_png}")
    # print(f"Saved PDF: {out_pdf}")


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
        "--show",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
