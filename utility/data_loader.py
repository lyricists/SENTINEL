import os
import pickle
import mat73
import numpy as np


def load_dataset(
    fPath,
    fileName,
    bPath=None,
    chName=None,
    t_end_ms=1000,
):
    """
    Load EEG dataset.

    Supports:
        selected good channels
        OR all channels

    Expected shape:
        channels x time x trials x subjects

    Assumes:
        epoch starts at −200 ms
        sampling rate = 250 Hz
    """

    with open(os.path.join(fPath, fileName), "rb") as f:
        Dataset = pickle.load(f)

    # ------------------------------------------------------------
    # Channel selection (optional)
    # ------------------------------------------------------------
    if chName is not None:

        goodCh = (
            mat73.loadmat(os.path.join(bPath, chName))["Channel"].astype(int).ravel()
            - 1
        )

        Dataset = Dataset[goodCh, :, :, :]

        print(f"Using selected channels: {len(goodCh)}")

    else:

        print(f"Using ALL channels: {Dataset.shape[0]}")

    # ------------------------------------------------------------
    # Crop time window
    # ------------------------------------------------------------
    sfreq = 250

    t = np.arange(Dataset.shape[1]) * (1000 / sfreq) - 200

    t_idx = np.where((t >= -200) & (t <= t_end_ms))[0]

    Dataset = Dataset[:, t_idx, :, :]

    print(f"Time window: −200 to {t_end_ms} ms")

    return Dataset


def load_subject_groups(bPath, subIdx):

    subject_group = mat73.loadmat(os.path.join(bPath, subIdx))["subject_index"].ravel()

    group_indices = {
        "Control": np.where(subject_group == 1)[0],
        "Depressed": np.where(subject_group == 2)[0],
        "Suicidal": np.where(subject_group == 3)[0],
    }

    return group_indices
