# ============================================================
# Extract trial-wise sentence ranking info per subject - parallel
# One output entry per EEG trial, not per log row
# ============================================================

import os
import json
import fnmatch
import pickle
import numpy as np
import mat73
from tqdm import tqdm
from joblib import Parallel, delayed

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

logPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/Log/"
save_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/"

ranking_file = os.path.join(
    save_path, "sentence_ranking_deepconvnet_val_balanced_accuracy.json"
)
logFile = "*sen*"

n_sub = 137
n_jobs = -1


# ------------------------------------------------------------
# Load ranking JSON
# ------------------------------------------------------------

with open(ranking_file, "r", encoding="utf-8") as f:
    ranking_results = json.load(f)


rank_lookup = {}

for comparison, rows in ranking_results.items():
    rank_lookup[comparison] = {}

    for row in rows:
        sentence = row["sentence"]
        rank_lookup[comparison][sentence] = int(row["rank"])

comparisons = list(rank_lookup.keys())


# ------------------------------------------------------------
# Load log files
# ------------------------------------------------------------

logfiles = sorted(
    [file for file in os.listdir(logPath) if fnmatch.fnmatch(file, logFile)]
)

assert len(logfiles) >= n_sub, "Number of log files is smaller than n_sub."


# ------------------------------------------------------------
# Parallel worker
# ------------------------------------------------------------


def process_subject(sub, logfile, logPath, rank_lookup, comparisons):

    dat_dict = mat73.loadmat(os.path.join(logPath, logfile))
    log_data = dat_dict["log"][1:]  # remove header

    trialIndex = np.array([element[0] for element in log_data]).astype(int) - 1
    sentenceLog = np.array([element[9] for element in log_data])
    conLog = np.array([element[12] for element in log_data])
    toiLog = np.array([element[14] for element in log_data])

    # --------------------------------------------------------
    # Find first row of each trial
    # Each trial has multiple word rows, so keep only trial starts
    # --------------------------------------------------------

    trial_start_rows = np.r_[0, np.flatnonzero(np.diff(trialIndex) != 0) + 1]

    sub_trial_info = {}

    for row_idx in trial_start_rows:

        trial_idx = int(trialIndex[row_idx])
        sentence = str(sentenceLog[row_idx])
        congruence = str(conLog[row_idx])
        toi = str(toiLog[row_idx])

        trial_dict = {
            "Sentence": sentence,
            "TOI": toi,
            "Congruence": congruence,
        }

        for comparison in comparisons:
            trial_dict[f"{comparison}_rank"] = rank_lookup[comparison].get(
                sentence, np.nan
            )

        sub_trial_info[trial_idx] = trial_dict

    return sub, sub_trial_info


# ------------------------------------------------------------
# Run in parallel
# ------------------------------------------------------------

results = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_subject)(
        sub,
        logfiles[sub],
        logPath,
        rank_lookup,
        comparisons,
    )
    for sub in tqdm(range(n_sub), desc="Extracting trial-level info")
)


# ------------------------------------------------------------
# Store as trialInfo[subject][trial_index]
# ------------------------------------------------------------

trialInfo = {
    sub: sub_trial_info for sub, sub_trial_info in sorted(results, key=lambda x: x[0])
}


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out_pkl = os.path.join(save_path, "trial_sentence_rank_info_dl.pkl")
out_json = os.path.join(save_path, "trial_sentence_rank_info_dl.json")

with open(out_pkl, "wb") as f:
    pickle.dump(trialInfo, f)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(trialInfo, f, indent=4, ensure_ascii=False)

print(f"Saved: {out_pkl}")
print(f"Saved: {out_json}")
