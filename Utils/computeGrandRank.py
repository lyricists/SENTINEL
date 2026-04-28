# ============================================================
# Compute grand sentence ranking across all comparisons
# ============================================================

import json
from collections import defaultdict


json_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/sentence_ranking.json"


with open(json_path, "r", encoding="utf-8") as f:
    ranking_results = json.load(f)


grand_scores = defaultdict(dict)

for comparison, sentences in ranking_results.items():

    for entry in sentences:

        sen_idx = entry["sentence_index"]

        if "sentence" not in grand_scores[sen_idx]:
            grand_scores[sen_idx]["sentence"] = entry["sentence"]
            grand_scores[sen_idx]["TOI"] = entry.get("TOI", None)
            grand_scores[sen_idx]["congruency"] = entry.get("congruency", None)

        if "values" not in grand_scores[sen_idx]:
            grand_scores[sen_idx]["values"] = {}

        grand_scores[sen_idx]["values"][comparison] = entry["value"]


grand_ranking = []

for sen_idx, info in grand_scores.items():

    total_value = sum(info["values"].values())

    grand_ranking.append(
        {
            "sentence_index": sen_idx,
            "sentence": info["sentence"],
            "TOI": info["TOI"],
            "congruency": info["congruency"],
            "Control_vs_Depressed": info["values"].get("Control_vs_Depressed", 0),
            "Control_vs_Suicidal": info["values"].get("Control_vs_Suicidal", 0),
            "Depressed_vs_Suicidal": info["values"].get("Depressed_vs_Suicidal", 0),
            "grand_value": total_value,
        }
    )


grand_ranking.sort(key=lambda x: x["grand_value"], reverse=True)

for i, entry in enumerate(grand_ranking):
    entry["grand_rank"] = i + 1


output_path = json_path.replace(".json", "_grand_rank.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(grand_ranking, f, indent=4, ensure_ascii=False)


print("Saved grand ranking to:")
print(output_path)
