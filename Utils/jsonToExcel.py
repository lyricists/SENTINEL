# ============================================================
# Convert sentence_ranking.json -> sentence_ranking.xlsx
# ============================================================

import os
import json
import pandas as pd


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

json_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/sentence_ranking.json"

excel_path = json_path.replace(".json", ".xlsx")


# ------------------------------------------------------------
# Load JSON
# ------------------------------------------------------------

with open(json_path, "r", encoding="utf-8") as f:
    ranking_results = json.load(f)


# ------------------------------------------------------------
# Convert to Excel (multi-sheet format)
# ------------------------------------------------------------

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:

    for comparison, rows in ranking_results.items():

        df = pd.DataFrame(rows)

        # Expand PC contributions into separate columns
        pc_cols = pd.DataFrame(
            df["pc_abs_diff"].tolist(),
            columns=[f"PC{i+1}" for i in range(len(df["pc_abs_diff"][0]))],
        )

        df = pd.concat([df.drop(columns="pc_abs_diff"), pc_cols], axis=1)

        df.to_excel(writer, sheet_name=comparison, index=False)


print("Excel file saved to:")
print(excel_path)
