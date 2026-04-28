# ============================================================
# Chunk-based sentence ranking summary
# ============================================================

import json
import os
from collections import defaultdict
import numpy as np


# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

json_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/sentence_ranking.json"

chunk_size = 40


# ------------------------------------------------------------
# LOAD JSON
# ------------------------------------------------------------

with open(json_path, "r") as f:
    ranking_results = json.load(f)


# ------------------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------------------


def chunk_sentences(sentence_list, chunk_size):

    return [
        sentence_list[i : i + chunk_size]
        for i in range(0, len(sentence_list), chunk_size)
    ]


# ------------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------------

chunk_summary = {}

for comparison, sentences in ranking_results.items():

    chunks = chunk_sentences(sentences, chunk_size)

    chunk_summary[comparison] = {}

    for i, chunk in enumerate(chunks):

        chunk_key = f"rank_{i*chunk_size+1}_{(i+1)*chunk_size}"

        toi_counter = defaultdict(int)
        congruency_counter = defaultdict(int)

        values = []
        pc_values = []

        for entry in chunk:

            # ranking score
            values.append(entry["value"])

            # PC contributions
            pc_values.append(entry["pc_abs_diff"])

            # TOI
            toi = entry["TOI"]
            toi_counter[toi] += 1
            congruency = entry["Congruence"]

            # congruency (if encoded inside TOI string)
            if congruency is not None:

                if "Congruent" in congruency:
                    congruency_counter["Congruent"] += 1

                elif "Incongruent" in congruency:
                    congruency_counter["Incongruent"] += 1

        pc_values = np.array(pc_values)

        chunk_summary[comparison][chunk_key] = {
            "n_sentences": len(chunk),
            "mean_value": float(np.mean(values)),
            "mean_PC": {
                f"PC{i+1}": float(np.mean(pc_values[:, i]))
                for i in range(pc_values.shape[1])
            },
            "TOI_counts": dict(toi_counter),
            "congruency_counts": dict(congruency_counter),
        }


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

output_path = json_path.replace(".json", "_chunk_summary.json")

with open(output_path, "w") as f:
    json.dump(chunk_summary, f, indent=4)


print("Saved chunk summary to:")
print(output_path)
