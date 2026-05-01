import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
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

from utility.bootstrap import (
    uniform_bootstrap_trials,
    congruence_contrast_bootstrap,
    sentence_response_average,
)
from utility.trial_selector import (
    get_rank_chunk_trials,
    split_trials_by_congruence,
)


def build_model(
    model_type,
    n_ch,
    n_time,
    n_classes=2,
    dropout=0.5,
):
    model_type = str(model_type).lower()

    if model_type == "deepconvnet":
        model = DeepConvNet(
            n_ch=n_ch,
            n_time=n_time,
            n_classes=n_classes,
            dropout=dropout,
        )

    elif model_type == "eegnet":
        model = EEGNet(
            n_ch=n_ch,
            n_time=n_time,
            n_classes=n_classes,
            F1=8,
            D=2,
            F2=16,
            kernel_length=64,
            dropout=dropout,
        )

    else:
        raise ValueError("model_type must be 'deepconvnet' or 'eegnet'.")

    return model


def resolve_device(device="auto"):
    device = str(device).lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    elif device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    elif device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"

    elif device == "cpu":
        return "cpu"

    else:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")


def save_learning_curve(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")

    if "val_balanced_acc" in history:
        plt.plot(
            epochs, history["val_balanced_acc"], label="Validation balanced accuracy"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Learning Curve - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_accuracy.png"), dpi=300)
    plt.close()


def evaluate_loader(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * xb.size(0)
            all_true.extend(yb.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)
    bacc = balanced_accuracy_score(all_true, all_pred)

    return avg_loss, acc, bacc


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    n_ch,
    n_time,
    n_classes=2,
    epochs=80,
    batch_size=64,
    lr=1e-3,
    dropout=0.5,
    device="auto",
    patience=10,
    min_delta=1e-4,
    early_stop_metric="val_balanced_accuracy",
    save_curve_path=None,
    model_type="deepconvnet",
    verbose=True,
    seed=42,
):
    device = resolve_device(device)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(
        model_type=model_type,
        n_ch=n_ch,
        n_time=n_time,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "train_balanced_acc": [],
        "val_acc": [],
        "val_balanced_acc": [],
    }

    if early_stop_metric == "val_loss":
        best_score = np.inf
        mode = "min"
    elif early_stop_metric == "val_balanced_accuracy":
        best_score = -np.inf
        mode = "max"
    else:
        raise ValueError(
            "early_stop_metric must be 'val_loss' or 'val_balanced_accuracy'"
        )

    best_state = None
    bad_epochs = 0
    best_epoch = 0

    epoch_iter = tqdm(range(1, epochs + 1), desc="Training", disable=not verbose)

    for epoch in epoch_iter:
        model.train()

        running_loss = 0.0
        train_true = []
        train_pred = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)

            running_loss += loss.item() * xb.size(0)
            train_true.extend(yb.cpu().numpy())
            train_pred.extend(pred.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_true, train_pred)
        train_bacc = balanced_accuracy_score(train_true, train_pred)

        val_loss, val_acc, val_bacc = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["train_balanced_acc"].append(float(train_bacc))
        history["val_acc"].append(float(val_acc))
        history["val_balanced_acc"].append(float(val_bacc))

        if early_stop_metric == "val_loss":
            current_score = val_loss
            improved = current_score < best_score - min_delta
        else:
            current_score = val_bacc
            improved = current_score > best_score + min_delta

        epoch_iter.set_postfix(
            {
                "tr_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "tr_bacc": f"{train_bacc:.3f}",
                "val_bacc": f"{val_bacc:.3f}",
            }
        )

        if improved:
            best_score = current_score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
            best_epoch = epoch
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch = {best_epoch}, "
                    f"best {early_stop_metric} = {best_score:.4f}"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = int(best_epoch)
    history["best_score"] = float(best_score)
    history["early_stop_metric"] = early_stop_metric

    if early_stop_metric == "val_loss":
        history["best_val_loss"] = float(best_score)
    else:
        history["best_val_balanced_accuracy"] = float(best_score)

    if save_curve_path is not None:
        save_learning_curve(history, save_curve_path)

    return model, history


def predict_subject_majority_vote(model, X_test_aug, device="auto"):
    device = resolve_device(device)

    model.eval()

    X_tensor = torch.tensor(X_test_aug, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)

    values, counts = np.unique(pred, return_counts=True)
    subject_pred = values[np.argmax(counts)]
    subject_prob_class1 = float(prob[:, 1].mean())

    return int(subject_pred), subject_prob_class1, pred, prob


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None

    return metrics


def make_augmented_subject_data(
    Dataset,
    trialInfo,
    sub,
    comparison,
    chunk_id,
    feature_mode,
    toi_mode,
    n_aug,
    k,
):
    trial_indices = get_rank_chunk_trials(
        trialInfo=trialInfo,
        sub=int(sub),
        comparison=comparison,
        chunk_id=chunk_id,
        toi_mode=toi_mode,
    )

    X_sub = Dataset[:, :, :, int(sub)]

    if feature_mode == "uniform":
        X_aug = uniform_bootstrap_trials(
            X=X_sub,
            trial_indices=trial_indices,
            n_aug=n_aug,
            k=k,
        )

    elif feature_mode == "contrast":
        con_idx, incon_idx = split_trials_by_congruence(
            trialInfo=trialInfo,
            sub=int(sub),
            trial_indices=trial_indices,
        )

        X_aug = congruence_contrast_bootstrap(
            X=X_sub,
            congruent_idx=con_idx,
            incongruent_idx=incon_idx,
            n_aug=n_aug,
            k=k,
        )

    elif feature_mode == "sentence_response":
        X_aug = sentence_response_average(
            X=X_sub,
            trialInfo=trialInfo,
            sub=int(sub),
            trial_indices=trial_indices,
        )

    else:
        raise ValueError(
            "feature_mode must be 'uniform', 'contrast', or 'sentence_response'."
        )

    return X_aug, trial_indices


def run_group_decoding_cv(
    Dataset,
    trialInfo,
    group_indices,
    comparison,
    group_a,
    group_b,
    chunk_id,
    feature_mode="uniform",
    toi_mode="all",
    model_type="eegnet",
    n_aug_train=200,
    n_aug_test=100,
    k=12,
    n_splits=5,
    epochs=80,
    batch_size=64,
    lr=1e-3,
    dropout=0.5,
    seed=42,
    device="auto",
    val_size=0.2,
    patience=10,
    min_delta=1e-4,
    early_stop_metric="val_balanced_accuracy",
    curve_dir=None,
    verbose=True,
):

    device = resolve_device(device)
    print(f"Using device: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    subjects = np.concatenate(
        [
            group_indices[group_a],
            group_indices[group_b],
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(len(group_indices[group_a]), dtype=int),
            np.ones(len(group_indices[group_b]), dtype=int),
        ]
    )

    n_ch, n_time, _, _ = Dataset.shape

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(
        skf.split(subjects, labels), start=1
    ):

        trainval_subjects = subjects[trainval_idx]
        test_subjects = subjects[test_idx]

        trainval_labels = labels[trainval_idx]
        test_labels = labels[test_idx]

        # --------------------------------------------------------
        # Subject-level train/validation split
        # --------------------------------------------------------
        train_idx, val_idx = train_test_split(
            np.arange(len(trainval_subjects)),
            test_size=val_size,
            stratify=trainval_labels,
            random_state=seed + fold,
        )

        train_subjects = trainval_subjects[train_idx]
        val_subjects = trainval_subjects[val_idx]

        train_labels = trainval_labels[train_idx]
        val_labels = trainval_labels[val_idx]

        # --------------------------------------------------------
        # Build augmented training data from train subjects only
        # --------------------------------------------------------
        X_train_all = []
        y_train_all = []

        skipped_train_subjects = []

        for sub, y in zip(train_subjects, train_labels):

            X_aug, trial_indices = make_augmented_subject_data(
                Dataset=Dataset,
                trialInfo=trialInfo,
                sub=int(sub),
                comparison=comparison,
                chunk_id=chunk_id,
                feature_mode=feature_mode,
                toi_mode=toi_mode,
                n_aug=n_aug_train,
                k=k,
            )

            if X_aug is None:
                skipped_train_subjects.append(int(sub))
                continue

            X_train_all.append(X_aug)
            y_train_all.append(np.full(X_aug.shape[0], int(y)))

        if len(X_train_all) == 0:
            raise RuntimeError("No training data was generated. Check trial selection.")

        X_train_all = np.concatenate(X_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        # --------------------------------------------------------
        # Build validation data from validation subjects only
        # --------------------------------------------------------
        X_val_all = []
        y_val_all = []

        skipped_val_subjects = []

        for sub, y in zip(val_subjects, val_labels):

            X_aug, trial_indices = make_augmented_subject_data(
                Dataset=Dataset,
                trialInfo=trialInfo,
                sub=int(sub),
                comparison=comparison,
                chunk_id=chunk_id,
                feature_mode=feature_mode,
                toi_mode=toi_mode,
                n_aug=n_aug_test,
                k=k,
            )

            if X_aug is None:
                skipped_val_subjects.append(int(sub))
                continue

            X_val_all.append(X_aug)
            y_val_all.append(np.full(X_aug.shape[0], int(y)))

        if len(X_val_all) == 0:
            raise RuntimeError(
                "No validation data was generated. Check trial selection."
            )

        X_val_all = np.concatenate(X_val_all, axis=0)
        y_val_all = np.concatenate(y_val_all, axis=0)

        # --------------------------------------------------------
        # Train model using explicit validation set
        # --------------------------------------------------------
        if curve_dir is not None:
            curve_path = os.path.join(
                curve_dir,
                f"{comparison}_chunk{chunk_id + 1}_{feature_mode}_{toi_mode}_{model_type}_fold{fold}.png",
            )
        else:
            curve_path = None

        model, train_history = train_model(
            X_train=X_train_all,
            y_train=y_train_all,
            X_val=X_val_all,
            y_val=y_val_all,
            n_ch=n_ch,
            n_time=n_time,
            n_classes=2,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            dropout=dropout,
            device=device,
            patience=patience,
            min_delta=min_delta,
            early_stop_metric=early_stop_metric,
            save_curve_path=curve_path,
            model_type=model_type,
            verbose=verbose,
            seed=seed + fold,
        )

        # --------------------------------------------------------
        # Test subject-level majority vote
        # --------------------------------------------------------
        subject_preds = []
        subject_probs = []
        subject_true = []
        subject_details = []
        skipped_test_subjects = []

        for sub, y in zip(test_subjects, test_labels):

            X_test_aug, trial_indices = make_augmented_subject_data(
                Dataset=Dataset,
                trialInfo=trialInfo,
                sub=int(sub),
                comparison=comparison,
                chunk_id=chunk_id,
                feature_mode=feature_mode,
                toi_mode=toi_mode,
                n_aug=n_aug_test,
                k=k,
            )

            if X_test_aug is None:
                skipped_test_subjects.append(int(sub))
                continue

            pred_subject, prob_subject, pred_aug, prob_aug = (
                predict_subject_majority_vote(
                    model=model,
                    X_test_aug=X_test_aug,
                    device=device,
                )
            )

            subject_preds.append(int(pred_subject))
            subject_probs.append(float(prob_subject))
            subject_true.append(int(y))

            subject_details.append(
                {
                    "subject": int(sub),
                    "true_label": int(y),
                    "pred_label": int(pred_subject),
                    "prob_class1": float(prob_subject),
                    "n_selected_trials": int(len(trial_indices)),
                    "selected_trials": [int(x) for x in trial_indices],
                    "augmented_predictions": pred_aug.astype(int).tolist(),
                }
            )

        metrics = compute_metrics(
            y_true=subject_true,
            y_pred=subject_preds,
            y_prob=subject_probs,
        )

        fold_result = {
            "fold": int(fold),
            "comparison": comparison,
            "group_a": group_a,
            "group_b": group_b,
            "chunk_id": int(chunk_id),
            "chunk_name": f"rank_{chunk_id * 40 + 1}_to_{(chunk_id + 1) * 40}",
            "feature_mode": feature_mode,
            "toi_mode": toi_mode,
            "metrics": metrics,
            "training_history": train_history,
            "y_true": [int(x) for x in subject_true],
            "y_pred": [int(x) for x in subject_preds],
            "y_prob_class1": [float(x) for x in subject_probs],
            "train_subjects": [int(x) for x in train_subjects],
            "val_subjects": [int(x) for x in val_subjects],
            "test_subjects": [int(x) for x in test_subjects],
            "skipped_train_subjects": skipped_train_subjects,
            "skipped_val_subjects": skipped_val_subjects,
            "skipped_test_subjects": skipped_test_subjects,
            "subject_details": subject_details,
            "model_type": model_type,
        }

        fold_results.append(fold_result)

        print(
            f"Fold {fold} | "
            f"{group_a} vs {group_b} | "
            f"chunk {chunk_id + 1} | "
            f"{feature_mode} | "
            f"{toi_mode} | "
            f"BACC={metrics['balanced_accuracy']:.3f}, "
            f"AUC={metrics['auc_roc']}, "
            f"F1={metrics['f1']:.3f}, "
            f"P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}, "
            f"BestEpoch={train_history['best_epoch']}"
        )

    return fold_results
