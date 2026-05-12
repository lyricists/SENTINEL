# ============================================================
# PCA-based Sentence rank analysis using Hedges' g
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
        goodCh = (
            mat73.loadmat(self.bPath + self.chName)["Channel"].astype(int).ravel() - 1
        )

        with open(self.fPath + self.fileName, "rb") as file:
            Dataset = pickle.load(file)

        Dataset = Dataset[goodCh, :, :, :]

        with open(self.bPath + self.senName, "r") as file:
            sen_list = list(json.load(file).keys())

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

        subject_to_group = {}
        for group_name, idxs in group_indices.items():
            for idx in idxs:
                subject_to_group[idx] = group_name

        for sub in range(self.n_sub):
            subject_mean = Dataset[:, :, :, sub].mean(axis=2)
            group_name = subject_to_group[sub]
            group_inputs[group_name].append(subject_mean)

        pca_input = np.concatenate(
            [
                np.mean(group_inputs[group_name], axis=0)
                for group_name in ["Control", "Depressed", "Suicidal"]
            ],
            axis=1,
        )

        pca = PCA(n_components=self.n_components)
        pca.fit(pca_input.T)

        X = Dataset.reshape(Dataset.shape[0], -1).T
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
    # Hedges' g
    # ------------------------------------------------------------
    @staticmethod
    def _hedges_g(x1, x2):
        """
        Compute signed Hedges' g between two groups.

        x1, x2: subject-level values for one PC and one sentence
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        x1 = x1[~np.isnan(x1)]
        x2 = x2[~np.isnan(x2)]

        n1 = len(x1)
        n2 = len(x2)

        if n1 < 2 or n2 < 2:
            return np.nan

        mean1 = np.mean(x1)
        mean2 = np.mean(x2)

        var1 = np.var(x1, ddof=1)
        var2 = np.var(x2, ddof=1)

        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_sd == 0 or np.isnan(pooled_sd):
            return np.nan

        cohen_d = (mean1 - mean2) / pooled_sd

        correction = 1 - (3 / (4 * (n1 + n2) - 9))

        hedges_g = correction * cohen_d

        return hedges_g

    # ------------------------------------------------------------
    # Sentence ranking using Hedges' g
    # ------------------------------------------------------------
    def rank_sentences(self, sentence_features, sen_list, senIdx, group_indices):
        """
        sentence_features: shape = (pc, sentence, subject)

        For each group pair:
        1. Compute signed Hedges' g for each PC and sentence.
        2. Take the absolute value of Hedges' g for each PC.
        3. Sum absolute Hedges' g values across PCs.
        4. Rank sentences in descending order.
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

            hedges_g_values = np.full(
                (self.n_components, len(sen_list)),
                np.nan,
                dtype=float,
            )

            for pc in range(self.n_components):
                for sen_idx in range(len(sen_list)):
                    x1 = sentence_features[pc, sen_idx, idx1]
                    x2 = sentence_features[pc, sen_idx, idx2]

                    hedges_g_values[pc, sen_idx] = self._hedges_g(x1, x2)

            # Final ranking score:
            # sum of absolute Hedges' g values across PCs
            score = np.nansum(np.abs(hedges_g_values), axis=0)

            order = np.argsort(score)[::-1]

            pair_key = f"{g1}_vs_{g2}"
            ranking_results[pair_key] = []

            for rank_idx, sen_idx in enumerate(order, start=1):
                toi_val = None
                con_val = None

                for sub in range(self.n_sub):
                    if senIdx[sub][sen_idx]["TOI"] is not None:
                        toi_val = senIdx[sub][sen_idx]["TOI"]
                        con_val = senIdx[sub][sen_idx]["Congruence"]
                        break

                ranking_results[pair_key].append(
                    {
                        "rank": int(rank_idx),
                        "sentence_index": int(sen_idx),
                        "sentence": sen_list[sen_idx],
                        "TOI": None if toi_val is None else str(toi_val),
                        "Congruence": None if con_val is None else str(con_val),
                        "value": float(score[sen_idx]),
                        "pc_abs_diff": hedges_g_values[:, sen_idx].tolist(),
                    }
                )

        return ranking_results

    # ------------------------------------------------------------
    # Save ranking
    # ------------------------------------------------------------
    def save_ranking_json(
        self, ranking_results, filename="sentence_ranking_hedges_g.json"
    ):
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

        sentence_features = np.full(
            (self.n_components, len(sen_list), self.n_sub),
            np.nan,
            dtype=float,
        )

        for n in tqdm(range(self.n_sub)):
            for k in range(len(sen_list)):
                idx = senIdx[n][k]["Index"]

                if idx.size == 0:
                    continue

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

        self.save_ranking_json(
            ranking_results,
            filename="sentence_ranking_hedges_g.json",
        )

        return sentence_features, ranking_results, senIdx


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    analyzer = SentenceRankAnalyzer(
        n_sub=137,
        n_components=3,
    )

    # sentence_features, ranking_results, senIdx = analyzer.run()
