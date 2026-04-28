# ============================================================
# PCA-based Sentence rank analysis
# ============================================================

import numpy as np
import pickle, json, os
from tqdm import tqdm
from sklearn.decomposition import PCA
import mat73
import fnmatch
from joblib import Parallel, delayed


class SentenceRankAnalyzer:
    def __init__(
        self,
        fPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
        bPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
        save_path: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/",
        logPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/Log/",
        fileName: str = "Data_sen_lepoch.pkl",
        logFile: str = "*sen*",
        chName: str = "GoodChannel.mat",
        subIdx: str = "subject_index.mat",
        senName: str = "final_sentiment.json",
        n_sub: int = 137,
        n_components: int = 5,
        tWin: list = [300, 600],
    ):
        self.fPath = fPath
        self.bPath = bPath
        self.logPath = logPath
        self.save_path = save_path
        self.fileName = fileName
        self.logFile = logFile
        self.chName = chName
        self.subIdx = subIdx
        self.senName = senName
        self.n_sub = n_sub
        self.n_components = n_components

        t = np.arange(-200, 1500, 4)
        self.tWin = np.where((t >= tWin[0]) & (t <= tWin[1]))[0]

    # ------------------------------------------------------------
    # Data load
    # ------------------------------------------------------------
    def load_data(self):
        # Channel index load
        goodCh = (
            mat73.loadmat(self.bPath + self.chName)["Channel"].astype(int).ravel() - 1
        )

        # EEG data load: expected shape = channels x time x trials x subjects
        with open(self.fPath + self.fileName, "rb") as file:
            Dataset = pickle.load(file)

        Dataset = Dataset[goodCh, :, :, :]

        # Sentence list
        with open(self.bPath + self.senName, "r") as file:
            sen_list = list(json.load(file).keys())

        # Subject group load
        subject_group = mat73.loadmat(self.bPath + self.subIdx)["subject_index"].ravel()

        group_indices = {
            "Control": np.where(subject_group == 1)[0],
            "Depressed": np.where(subject_group == 2)[0],
            "Suicidal": np.where(subject_group == 3)[0],
        }

        return Dataset, group_indices, sen_list

    # ------------------------------------------------------------
    # PCA spatial feature extraction
    # ------------------------------------------------------------
    def pca_extraction(self, Dataset, group_indices):
        group_inputs = {"Control": [], "Depressed": [], "Suicidal": []}

        # Faster subject -> group mapping
        subject_to_group = {}
        for group_name, idxs in group_indices.items():
            for idx in idxs:
                subject_to_group[idx] = group_name

        for sub in range(self.n_sub):
            subject_mean = Dataset[:, :, :, sub].mean(axis=2)  # ch x time
            group_name = subject_to_group[sub]
            group_inputs[group_name].append(subject_mean)

        pca_input = np.concatenate(
            [
                np.mean(group_inputs[group_name], axis=0)
                for group_name in ["Control", "Depressed", "Suicidal"]
            ],
            axis=1,
        )  # ch x (time * 3)

        pca = PCA(n_components=self.n_components)
        pca.fit(pca_input.T)  # samples x channels

        X = Dataset.reshape(Dataset.shape[0], -1).T  # (time*trials*subjects) x ch
        X_pca = pca.transform(X)

        Dataset_pca = X_pca.T.reshape(
            self.n_components,
            Dataset.shape[1],
            Dataset.shape[2],
            Dataset.shape[3],
        )

        return Dataset_pca

    # ------------------------------------------------------------
    # Helper for parallel sentence index extraction
    # ------------------------------------------------------------
    @staticmethod
    def _process_subject(log_path, log_file, sen_list, n):
        dat_dict = mat73.loadmat(os.path.join(log_path, log_file))
        trialIndex = np.array([element[0] for element in dat_dict["log"][1:]])
        sentenceLog = np.array([element[9] for element in dat_dict["log"][1:]])
        toiLog = np.array([element[14] for element in dat_dict["log"][1:]])
        conLog = np.array([element[12] for element in dat_dict["log"][1:]])

        sub_result = {}

        for k, s in enumerate(sen_list):
            Id = np.flatnonzero(sentenceLog == s)

            if Id.size == 0:
                sub_result[k] = {
                    "Index": np.array([], dtype=int),
                    "Sentence": s,
                    "TOI": None,
                    "Congruence": None,
                }
                continue

            start_indices = np.r_[0, np.flatnonzero(np.diff(Id) != 1) + 1]

            sub_result[k] = {
                "Index": trialIndex[Id[start_indices]].astype(int) - 1,
                "Sentence": s,
                "TOI": toiLog[Id[0]],
                "Congruence": conLog[Id[0]],
            }

        return n, sub_result

    def sentenceIdx(self, sen_list, n_jobs=-1):
        logfile = sorted(
            [
                logfile
                for logfile in os.listdir(self.logPath)
                if fnmatch.fnmatch(logfile, self.logFile)
            ]
        )

        print("Preparing index data...")

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._process_subject)(self.logPath, logfile[n], sen_list, n)
            for n in tqdm(range(self.n_sub))
        )

        senIdx = {n: sub_result for n, sub_result in results}
        return senIdx

    # ------------------------------------------------------------
    # Sentence ranking
    # ------------------------------------------------------------
    def rank_sentences(self, sentence_features, sen_list, senIdx, group_indices):
        """
        sentence_features: shape = (pc, sentence, subject)

        For each group pair:
        1. Mean within each group for each PC and sentence
        2. Absolute difference between the two groups for each PC
        3. Sum across PCs
        4. Rank sentences in descending order
        """
        group_pairs = [
            ("Control", "Depressed"),
            ("Control", "Suicidal"),
            ("Depressed", "Suicidal"),
        ]

        ranking_results = {}

        for g1, g2 in group_pairs:
            idx1 = group_indices[g1]
            idx2 = group_indices[g2]

            # group means: (pc, sentence)
            g1_mean = np.nanmean(sentence_features[:, :, idx1], axis=2)
            g2_mean = np.nanmean(sentence_features[:, :, idx2], axis=2)

            # abs difference per pc: (pc, sentence)
            abs_diff = np.abs(g1_mean - g2_mean)

            # summed value across PCs: (sentence,)
            score = np.nansum(abs_diff, axis=0)

            # ranking: highest score = rank 1
            order = np.argsort(score)[::-1]

            pair_key = f"{g1}_vs_{g2}"
            ranking_results[pair_key] = []

            for rank_idx, sen_idx in enumerate(order, start=1):
                # get TOI from first available subject for this sentence
                toi_val = None
                for sub in range(self.n_sub):
                    if senIdx[sub][sen_idx]["TOI"] is not None:
                        toi_val = senIdx[sub][sen_idx]["TOI"]
                        break

                ranking_results[pair_key].append(
                    {
                        "rank": int(rank_idx),
                        "sentence_index": int(sen_idx),
                        "sentence": sen_list[sen_idx],
                        "TOI": None if toi_val is None else str(toi_val),
                        "Congruence": str(senIdx[sub][sen_idx]["Congruence"]),
                        "value": float(score[sen_idx]),
                        "pc_abs_diff": abs_diff[:, sen_idx].tolist(),
                    }
                )

        return ranking_results

    # ------------------------------------------------------------
    # Save ranking
    # ------------------------------------------------------------
    def save_ranking_json(self, ranking_results, filename="sentence_ranking.json"):
        os.makedirs(self.save_path, exist_ok=True)
        out_path = os.path.join(self.save_path, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ranking_results, f, indent=4, ensure_ascii=False)

        print(f"Saved ranking results to: {out_path}")

    # ------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------
    def run(self):
        print("Processing loading data...")

        Dataset, group_indices, sen_list = self.load_data()
        Dataset_pca = self.pca_extraction(Dataset, group_indices)
        senIdx = self.sentenceIdx(sen_list, n_jobs=-1)

        print("Processing Sentence evaluation...")

        # Use NaN so missing sentences do not bias group means
        sentence_features = np.full(
            (self.n_components, len(sen_list), self.n_sub),
            np.nan,
            dtype=float,
        )  # n_components x sentences x subjects

        for n in tqdm(range(self.n_sub)):
            for k in range(len(sen_list)):
                idx = senIdx[n][k]["Index"]

                if idx.size == 0:
                    continue

                # Dataset_pca[:, :, idx, n] -> (pc, time, trial)
                # average over selected trials and time window
                sentence_features[:, k, n] = np.mean(
                    np.mean(Dataset_pca[:, :, idx, n][:, self.tWin, :], axis=2),
                    axis=1,
                )

        ranking_results = self.rank_sentences(
            sentence_features=sentence_features,
            sen_list=sen_list,
            senIdx=senIdx,
            group_indices=group_indices,
        )

        self.save_ranking_json(ranking_results, filename="sentence_ranking.json")

        return sentence_features, ranking_results, senIdx


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    analyzer = SentenceRankAnalyzer(
        n_sub=137,
        n_components=3,
    )
    sentence_features, ranking_results, senIdx = analyzer.run()
