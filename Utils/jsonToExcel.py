# ============================================================
# Convert sentence ranking JSON -> Excel
# Works for:
#   1) sentence_ranking.json
#   2) sentence_ranking_grand_rank.json
# ============================================================

import json
import pandas as pd


json_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/sentence_ranking_grand_rank.json"
excel_path = json_path.replace(".json", ".xlsx")


with open(json_path, "r", encoding="utf-8") as f:
    ranking_results = json.load(f)


def expand_pc_columns(df):
    """Expand pc_abs_diff column into PC1, PC2, PC3... if it exists."""

    if "pc_abs_diff" not in df.columns:
        return df

    if df["pc_abs_diff"].isnull().all():
        return df.drop(columns="pc_abs_diff")

    pc_cols = pd.DataFrame(
        df["pc_abs_diff"].tolist(),
        columns=[f"PC{i+1}" for i in range(len(df["pc_abs_diff"].iloc[0]))],
    )

    df = pd.concat([df.drop(columns="pc_abs_diff"), pc_cols], axis=1)

    return df


with pd.ExcelWriter(excel_path) as writer:

    # Case 1: sentence_ranking.json
    # Structure: {"Control_vs_Depressed": [...], ...}
    if isinstance(ranking_results, dict):

        for comparison, rows in ranking_results.items():

            df = pd.DataFrame(rows)
            df = expand_pc_columns(df)

            sheet_name = comparison[:31]  # Excel sheet-name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Case 2: sentence_ranking_grand_rank.json
    # Structure: [{...}, {...}, ...]
    elif isinstance(ranking_results, list):

        df = pd.DataFrame(ranking_results)
        df = expand_pc_columns(df)

        df.to_excel(writer, sheet_name="Grand_Ranking", index=False)

    else:
        raise ValueError("Unsupported JSON structure.")


print("Excel file saved to:")
print(excel_path)
