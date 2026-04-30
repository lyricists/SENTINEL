import pickle
import numpy as np


BIO_LABELS = [
    "bio",
    "biographical",
    "neutral biographical",
    "neutral_biographical",
]


def load_trial_info(path):
    with open(path, "rb") as f:
        trialInfo = pickle.load(f)

    return trialInfo


def is_bio_toi(toi):
    return str(toi).strip().lower() in BIO_LABELS


def add_non_bio_ranks(trialInfo, comparisons):
    """
    Uses existing trialInfo only.

    For each comparison:
        original rank: 1-160
        exclude Bio trials
        sort remaining unique sentences by original rank
        assign non-Bio rank: 1-120
    """

    for comparison in comparisons:
        rank_key = f"{comparison}_rank"
        non_bio_key = f"{comparison}_non_bio_rank"

        sentence_rank = {}

        for sub in trialInfo:
            for trial_idx, info in trialInfo[sub].items():

                if is_bio_toi(info["TOI"]):
                    continue

                sentence = info["Sentence"]
                rank = info.get(rank_key, np.nan)

                if rank is None or np.isnan(rank):
                    continue

                if sentence not in sentence_rank:
                    sentence_rank[sentence] = rank

        sorted_sentences = sorted(sentence_rank.keys(), key=lambda s: sentence_rank[s])

        non_bio_lookup = {
            sentence: new_rank
            for new_rank, sentence in enumerate(sorted_sentences, start=1)
        }

        for sub in trialInfo:
            for trial_idx, info in trialInfo[sub].items():
                sentence = info["Sentence"]

                if sentence in non_bio_lookup:
                    info[non_bio_key] = non_bio_lookup[sentence]
                else:
                    info[non_bio_key] = np.nan

    return trialInfo


def get_rank_chunk_trials(
    trialInfo,
    sub,
    comparison,
    chunk_id,
    chunk_size=40,
    toi_mode="all",
):
    rank_min = chunk_id * chunk_size + 1
    rank_max = (chunk_id + 1) * chunk_size

    if toi_mode == "all":
        rank_key = f"{comparison}_rank"
    elif toi_mode == "non_bio":
        rank_key = f"{comparison}_non_bio_rank"
    else:
        raise ValueError("toi_mode must be 'all' or 'non_bio'.")

    selected_trials = []

    for trial_idx, info in trialInfo[sub].items():

        if toi_mode == "non_bio" and is_bio_toi(info["TOI"]):
            continue

        rank = info.get(rank_key, np.nan)

        if rank is None or np.isnan(rank):
            continue

        if rank_min <= rank <= rank_max:
            selected_trials.append(int(trial_idx))

    return sorted(selected_trials)


def split_trials_by_congruence(trialInfo, sub, trial_indices):
    congruent = []
    incongruent = []

    for idx in trial_indices:
        con = str(trialInfo[sub][idx]["Congruence"]).lower()

        if "incon" in con:
            incongruent.append(int(idx))
        else:
            congruent.append(int(idx))

    return congruent, incongruent
