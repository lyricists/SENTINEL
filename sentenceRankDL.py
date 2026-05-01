# ============================================================
# Deep-learning-based Sentence rank analysis
# EEGNet / DeepConvNet
# Subject-level CV + subject-level validation split
# Output format matches previous sentence_ranking.json
#
# Early stopping options:
#   early_stop_metric="val_loss"
#   early_stop_metric="val_balanced_accuracy"
# ============================================================

import os
import json
import pickle
import fnmatch
import numpy as np
import mat73
import torch
import torch.nn as nn

from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from model.deepconvnet import DeepConvNet
from model.eegnet import EEGNet


class SentenceNNRankAnalyzer:
    def __init__(
        self,
        fPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
        bPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
        save_path: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/SENTINEL/Results/",
        logPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/Log/",
        fileName: str = "Data_sen_lepoch.pkl",
        logFile: str = "*sen*",
        subIdx: str = "subject_index.mat",
        senName: str = "final_sentiment.json",
        n_sub: int = 137,
        model_name: str = "eegnet",
        n_splits: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_epochs: int = 80,
        patience: int = 10,
        val_size: float = 0.1,
        min_delta: float = 1e-4,
        early_stop_metric: str = "val_balanced_accuracy",
        random_state: int = 42,
        device: str = "auto",
    ):
        self.fPath = fPath
        self.bPath = bPath
        self.save_path = save_path
        self.logPath = logPath
        self.fileName = fileName
        self.logFile = logFile
        self.subIdx = subIdx
        self.senName = senName
        self.n_sub = n_sub

        self.model_name = model_name.lower()
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.min_delta = min_delta
        self.early_stop_metric = early_stop_metric
        self.random_state = random_state
        self.device = self.resolve_device(device)

        if self.early_stop_metric not in ["val_loss", "val_balanced_accuracy"]:
            raise ValueError(
                "early_stop_metric must be 'val_loss' or 'val_balanced_accuracy'"
            )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    @staticmethod
    def resolve_device(device="auto"):
        device = str(device).lower()

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        if device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"

        return "cpu"

    # ------------------------------------------------------------
    # Data load
    # ------------------------------------------------------------

    def load_data(self):
        """
        Expected Dataset shape:
            channel x time x trial x subject

        This version uses all channels.
        No good-channel selection is applied.
        """

        with open(os.path.join(self.fPath, self.fileName), "rb") as file:
            Dataset = pickle.load(file)

        Dataset = np.asarray(Dataset, dtype=np.float32)

        Dataset = Dataset[
            :, np.arange(300), :, :
        ]  # Crop to 300 time points (−200 to 1000 ms at 250 Hz)

        with open(os.path.join(self.bPath, self.senName), "r") as file:
            sen_list = list(json.load(file).keys())

        subject_group = mat73.loadmat(os.path.join(self.bPath, self.subIdx))[
            "subject_index"
        ].ravel()

        group_indices = {
            "Control": np.where(subject_group == 1)[0],
            "Depressed": np.where(subject_group == 2)[0],
            "Suicidal": np.where(subject_group == 3)[0],
        }

        return Dataset, group_indices, sen_list

    # ------------------------------------------------------------
    # Sentence index extraction from randomized log files
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
            matched_trials = trialIndex[Id[start_indices]].astype(int) - 1

            sub_result[k] = {
                "Index": matched_trials,
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

        if len(logfile) < self.n_sub:
            raise ValueError(
                f"Found only {len(logfile)} log files, but n_sub={self.n_sub}."
            )

        print("Preparing sentence index data from log files...")

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._process_subject)(
                self.logPath,
                logfile[n],
                sen_list,
                n,
            )
            for n in tqdm(range(self.n_sub))
        )

        senIdx = {n: sub_result for n, sub_result in results}
        return senIdx

    # ------------------------------------------------------------
    # Sanity check for sentence-trial matching
    # ------------------------------------------------------------

    def check_sentence_trial_matching(self, senIdx, sen_list):
        n_missing = 0
        n_single = 0
        n_two = 0
        n_more_than_two = 0

        for sub in range(self.n_sub):
            for sen in range(len(sen_list)):
                idx = senIdx[sub][sen]["Index"]

                if idx.size == 0:
                    n_missing += 1
                elif idx.size == 1:
                    n_single += 1
                elif idx.size == 2:
                    n_two += 1
                else:
                    n_more_than_two += 1

        print("\nSentence-trial matching check")
        print("--------------------------------")
        print(f"Missing sentence entries : {n_missing}")
        print(f"Single-trial sentences   : {n_single}")
        print(f"Two-trial sentences      : {n_two}")
        print(f">2-trial sentences       : {n_more_than_two}")
        print("--------------------------------\n")

    # ------------------------------------------------------------
    # Sentence response averaging
    # ------------------------------------------------------------

    def make_sentence_response(self, Dataset, senIdx, sen_list):
        """
        Convert trial-level EEG into sentence-level responses.

        Input:
            Dataset: channel x time x trial x subject

        Output:
            sentence_data: subject x sentence x channel x time
        """

        n_ch, n_time, _, n_sub = Dataset.shape
        n_sentence = len(sen_list)

        sentence_data = np.full(
            (n_sub, n_sentence, n_ch, n_time),
            np.nan,
            dtype=np.float32,
        )

        for sub in tqdm(range(n_sub), desc="Averaging trials per sentence"):
            for sen in range(n_sentence):
                idx = senIdx[sub][sen]["Index"]

                if idx.size == 0:
                    continue

                sentence_data[sub, sen] = np.nanmean(
                    Dataset[:, :, idx, sub],
                    axis=2,
                )

        return sentence_data

    # ------------------------------------------------------------
    # Group-pair selection
    # ------------------------------------------------------------

    @staticmethod
    def select_pair(sentence_data, group_indices, group_a, group_b):
        idx_a = group_indices[group_a]
        idx_b = group_indices[group_b]

        selected_idx = np.concatenate([idx_a, idx_b])

        X = sentence_data[selected_idx]

        y = np.concatenate(
            [
                np.zeros(len(idx_a), dtype=np.int64),
                np.ones(len(idx_b), dtype=np.int64),
            ]
        )

        return X, y, selected_idx

    # ------------------------------------------------------------
    # Subject x sentence -> sentence sample-level data
    # ------------------------------------------------------------

    @staticmethod
    def subject_sentence_to_samples(X, y):
        """
        X:
            subject x sentence x channel x time

        Output:
            X_sample:
                sample x 1 x channel x time
        """

        n_sub, n_sentence, n_ch, n_time = X.shape

        X_sample = X.reshape(n_sub * n_sentence, n_ch, n_time)
        X_sample = X_sample[:, None, :, :]

        subject_ref = np.repeat(np.arange(n_sub), n_sentence)
        sentence_ref = np.tile(np.arange(n_sentence), n_sub)

        y_sample = y[subject_ref]

        valid = ~np.isnan(X_sample).any(axis=(1, 2, 3))

        return (
            X_sample[valid],
            y_sample[valid],
            subject_ref[valid],
            sentence_ref[valid],
        )

    # ------------------------------------------------------------
    # Model builder
    # ------------------------------------------------------------

    def build_model(self, n_chans, n_times):
        if self.model_name == "deepconvnet":
            return DeepConvNet(
                n_ch=n_chans,
                n_time=n_times,
                n_classes=2,
            )

        elif self.model_name == "eegnet":
            return EEGNet(
                n_ch=n_chans,
                n_time=n_times,
                n_classes=2,
            )

        else:
            raise ValueError("model_name must be 'eegnet' or 'deepconvnet'")

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def train_model(self, model, X_train, y_train, X_val, y_val):
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        if self.early_stop_metric == "val_loss":
            best_score = np.inf
        else:
            best_score = -np.inf

        best_state = None
        best_epoch = 0
        wait = 0

        for epoch in range(self.max_epochs):
            # --------------------------------------------------------
            # Training
            # --------------------------------------------------------
            model.train()

            train_loss = 0.0
            train_true = []
            train_pred = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()

                logits = model(xb)
                loss = criterion(logits, yb)

                loss.backward()
                optimizer.step()

                pred = torch.argmax(logits, dim=1)

                train_loss += loss.item() * len(yb)
                train_true.extend(yb.cpu().numpy())
                train_pred.extend(pred.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_true, train_pred)
            train_bacc = balanced_accuracy_score(train_true, train_pred)

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------
            model.eval()

            val_loss = 0.0
            val_true = []
            val_pred = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    logits = model(xb)
                    loss = criterion(logits, yb)
                    pred = torch.argmax(logits, dim=1)

                    val_loss += loss.item() * len(yb)
                    val_true.extend(yb.cpu().numpy())
                    val_pred.extend(pred.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(val_true, val_pred)
            val_bacc = balanced_accuracy_score(val_true, val_pred)

            # --------------------------------------------------------
            # Early stopping criterion
            # --------------------------------------------------------
            if self.early_stop_metric == "val_loss":
                current_score = val_loss
                improved = current_score < best_score - self.min_delta
            else:
                current_score = val_bacc
                improved = current_score > best_score + self.min_delta

            if improved:
                best_score = current_score
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1

            print(
                f"Epoch {epoch + 1:03d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"train_bacc={train_bacc:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_bacc={val_bacc:.4f} | "
                f"best_{self.early_stop_metric}={best_score:.4f} | "
                f"best_epoch={best_epoch} | "
                f"wait={wait}/{self.patience}"
            )

            if wait >= self.patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch = {best_epoch}, "
                    f"best {self.early_stop_metric} = {best_score:.4f}"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    def predict_model(self, model, X):
        loader = DataLoader(
            torch.tensor(X, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False,
        )

        model.eval()
        probs = []

        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=1)
                probs.append(prob.cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        preds = np.argmax(probs, axis=1)

        return preds, probs

    # ------------------------------------------------------------
    # Metadata helper
    # ------------------------------------------------------------

    def get_sentence_metadata(self, senIdx, sen_idx):
        toi_val = None
        con_val = None

        for sub in range(self.n_sub):
            if senIdx[sub][sen_idx]["TOI"] is not None:
                toi_val = senIdx[sub][sen_idx]["TOI"]
                con_val = senIdx[sub][sen_idx]["Congruence"]
                break

        return toi_val, con_val

    # ------------------------------------------------------------
    # Pairwise ranking
    # ------------------------------------------------------------

    def rank_pairwise(
        self,
        sentence_data,
        sen_list,
        senIdx,
        group_indices,
        group_a,
        group_b,
    ):
        X, y, original_subject_idx = self.select_pair(
            sentence_data,
            group_indices,
            group_a,
            group_b,
        )

        n_sub, n_sentence, n_ch, n_time = X.shape

        sentence_correct = np.full((n_sub, n_sentence), np.nan)
        sentence_pred = np.full((n_sub, n_sentence), np.nan)
        sentence_prob = np.full((n_sub, n_sentence), np.nan)

        subject_true = []
        subject_pred = []
        subject_prob = []
        subject_original = []

        fold_metrics = []

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"{group_a} vs {group_b} | fold {fold + 1}/{self.n_splits}")

            # Subject-level train/validation split
            train_sub_idx, val_sub_idx = train_test_split(
                train_idx,
                test_size=self.val_size,
                stratify=y[train_idx],
                random_state=self.random_state + fold,
            )

            X_train_sub = X[train_sub_idx]
            y_train_sub = y[train_sub_idx]

            X_val_sub = X[val_sub_idx]
            y_val_sub = y[val_sub_idx]

            X_test_sub = X[test_idx]
            y_test_sub = y[test_idx]

            X_train_all, y_train_all, _, _ = self.subject_sentence_to_samples(
                X_train_sub,
                y_train_sub,
            )

            X_val_all, y_val_all, _, _ = self.subject_sentence_to_samples(
                X_val_sub,
                y_val_sub,
            )

            (
                X_test_all,
                y_test_all,
                test_subject_ref,
                test_sentence_ref,
            ) = self.subject_sentence_to_samples(
                X_test_sub,
                y_test_sub,
            )

            model = self.build_model(
                n_chans=n_ch,
                n_times=n_time,
            )

            model = self.train_model(
                model=model,
                X_train=X_train_all,
                y_train=y_train_all,
                X_val=X_val_all,
                y_val=y_val_all,
            )

            pred_all, prob_all = self.predict_model(model, X_test_all)

            print(
                "Test prediction counts:",
                np.bincount(pred_all, minlength=2),
                "| True counts:",
                np.bincount(y_test_all, minlength=2),
            )

            # Store held-out sentence-level predictions
            for i in range(len(pred_all)):
                local_sub = test_subject_ref[i]
                sen = test_sentence_ref[i]

                pairwise_sub = test_idx[local_sub]

                pred = int(pred_all[i])
                true = int(y_test_all[i])

                sentence_pred[pairwise_sub, sen] = pred
                sentence_prob[pairwise_sub, sen] = float(prob_all[i, 1])
                sentence_correct[pairwise_sub, sen] = int(pred == true)

            # Subject-level majority vote
            fold_subject_true = []
            fold_subject_pred = []

            for local_sub in range(len(test_idx)):
                pairwise_sub = test_idx[local_sub]

                preds = sentence_pred[pairwise_sub]
                probs = sentence_prob[pairwise_sub]

                valid = ~np.isnan(preds)

                if np.sum(valid) == 0:
                    continue

                final_pred = int(np.nanmean(preds[valid]) >= 0.5)
                final_prob = float(np.nanmean(probs[valid]))

                subject_true.append(int(y[pairwise_sub]))
                subject_pred.append(final_pred)
                subject_prob.append(final_prob)
                subject_original.append(int(original_subject_idx[pairwise_sub]))

                fold_subject_true.append(int(y[pairwise_sub]))
                fold_subject_pred.append(final_pred)

            fold_metrics.append(
                {
                    "fold": int(fold + 1),
                    "n_train_subjects": int(len(train_sub_idx)),
                    "n_val_subjects": int(len(val_sub_idx)),
                    "n_test_subjects": int(len(test_idx)),
                    "accuracy": float(
                        accuracy_score(fold_subject_true, fold_subject_pred)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            fold_subject_true,
                            fold_subject_pred,
                        )
                    ),
                }
            )

        # --------------------------------------------------------
        # Overall subject-level metrics
        # --------------------------------------------------------

        subject_metrics = {
            "accuracy": float(accuracy_score(subject_true, subject_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(subject_true, subject_pred)
            ),
            "f1": float(f1_score(subject_true, subject_pred, zero_division=0)),
            "precision": float(
                precision_score(subject_true, subject_pred, zero_division=0)
            ),
            "recall": float(recall_score(subject_true, subject_pred, zero_division=0)),
        }

        try:
            subject_metrics["auc"] = float(roc_auc_score(subject_true, subject_prob))
        except ValueError:
            subject_metrics["auc"] = None

        # --------------------------------------------------------
        # Sentence-level ranking
        # Main ranking value = sentence balanced accuracy
        # --------------------------------------------------------

        sentence_ranking = []

        for sen in range(n_sentence):
            valid = ~np.isnan(sentence_correct[:, sen])

            if np.sum(valid) == 0:
                continue

            y_valid = y[valid]
            pred_valid = sentence_pred[valid, sen].astype(int)
            correct_valid = sentence_correct[valid, sen]

            sent_acc = float(np.mean(correct_valid))

            try:
                sent_bal_acc = float(balanced_accuracy_score(y_valid, pred_valid))
            except ValueError:
                sent_bal_acc = None

            group_a_mask = (y == 0) & valid
            group_b_mask = (y == 1) & valid

            group_a_acc = (
                float(np.mean(sentence_correct[group_a_mask, sen]))
                if np.sum(group_a_mask) > 0
                else None
            )

            group_b_acc = (
                float(np.mean(sentence_correct[group_b_mask, sen]))
                if np.sum(group_b_mask) > 0
                else None
            )

            toi_val, con_val = self.get_sentence_metadata(senIdx, sen)

            sentence_ranking.append(
                {
                    "rank": None,
                    "sentence_index": int(sen),
                    "sentence": sen_list[sen],
                    "TOI": None if toi_val is None else str(toi_val),
                    "Congruence": None if con_val is None else str(con_val),
                    "value": sent_bal_acc,
                    "accuracy": sent_acc,
                    "balanced_accuracy": sent_bal_acc,
                    f"{group_a}_accuracy": group_a_acc,
                    f"{group_b}_accuracy": group_b_acc,
                    "n_subjects": int(np.sum(valid)),
                    "mean_prob_class_1": float(np.nanmean(sentence_prob[valid, sen])),
                }
            )

        sentence_ranking = sorted(
            sentence_ranking,
            key=lambda x: -np.inf if x["value"] is None else x["value"],
            reverse=True,
        )

        for rank_idx, row in enumerate(sentence_ranking, start=1):
            row["rank"] = int(rank_idx)

        metrics_result = {
            "comparison": f"{group_a}_vs_{group_b}",
            "model": self.model_name,
            "label_mapping": {
                group_a: 0,
                group_b: 1,
            },
            "n_subjects": int(n_sub),
            "n_sentences": int(n_sentence),
            "n_channels": int(n_ch),
            "n_times": int(n_time),
            "n_splits": int(self.n_splits),
            "val_size": float(self.val_size),
            "early_stop_metric": self.early_stop_metric,
            "subject_metrics": subject_metrics,
            "fold_metrics": fold_metrics,
            "subject_predictions": [
                {
                    "subject_index": int(subject_original[i]),
                    "true_label": int(subject_true[i]),
                    "pred_label": int(subject_pred[i]),
                    "mean_prob_class_1": float(subject_prob[i]),
                }
                for i in range(len(subject_true))
            ],
        }

        return sentence_ranking, metrics_result

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------

    def save_json(self, obj, filename):
        os.makedirs(self.save_path, exist_ok=True)
        out_path = os.path.join(self.save_path, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)

        print(f"Saved: {out_path}")

    # ------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------

    def run(self):
        print(f"Device: {self.device}")
        print(f"Early stopping metric: {self.early_stop_metric}")
        print(f"Validation subject split size: {self.val_size}")
        print("Loading data...")

        Dataset, group_indices, sen_list = self.load_data()

        print(f"Dataset shape: {Dataset.shape}")
        print("Expected shape: channel x time x trial x subject")
        print("Using all channels. No good-channel filtering.")

        senIdx = self.sentenceIdx(sen_list, n_jobs=-1)
        self.check_sentence_trial_matching(senIdx, sen_list)

        print("Creating sentence-level EEG responses...")
        sentence_data = self.make_sentence_response(
            Dataset=Dataset,
            senIdx=senIdx,
            sen_list=sen_list,
        )

        group_pairs = [
            ("Control", "Depressed"),
            ("Control", "Suicidal"),
            ("Depressed", "Suicidal"),
        ]

        ranking_results = {}
        metrics_results = {}

        for group_a, group_b in group_pairs:
            pair_key = f"{group_a}_vs_{group_b}"

            sentence_ranking, metrics_result = self.rank_pairwise(
                sentence_data=sentence_data,
                sen_list=sen_list,
                senIdx=senIdx,
                group_indices=group_indices,
                group_a=group_a,
                group_b=group_b,
            )

            ranking_results[pair_key] = sentence_ranking
            metrics_results[pair_key] = metrics_result

        ranking_filename = (
            f"sentence_ranking_{self.model_name}_{self.early_stop_metric}.json"
        )
        metrics_filename = (
            f"sentence_ranking_{self.model_name}_{self.early_stop_metric}_metrics.json"
        )

        self.save_json(ranking_results, ranking_filename)
        self.save_json(metrics_results, metrics_filename)

        return sentence_data, ranking_results, metrics_results, senIdx


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    analyzer = SentenceNNRankAnalyzer(
        n_sub=137,
        model_name="deepconvnet",  # "eegnet" or "deepconvnet"
        n_splits=5,
        batch_size=64,
        lr=1e-3,
        max_epochs=40,
        patience=10,
        val_size=0.1,
        min_delta=1e-4,
        early_stop_metric="val_balanced_accuracy",  # "val_loss" or "val_balanced_accuracy"
        random_state=42,
        device="auto",
    )

    sentence_data, ranking_results, metrics_results, senIdx = analyzer.run()
