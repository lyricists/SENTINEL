import os
import pickle
import mat73
import numpy as np


def load_dataset(fPath, fileName):
    with open(os.path.join(fPath, fileName), "rb") as f:
        Dataset = pickle.load(f)

    # Use all channels.
    # Expected shape: ch x time x trial x subject
    return Dataset[:, np.arange(300), :, :]


def load_subject_groups(bPath, subIdx):
    subject_group = mat73.loadmat(os.path.join(bPath, subIdx))["subject_index"].ravel()

    group_indices = {
        "Control": np.where(subject_group == 1)[0],
        "Depressed": np.where(subject_group == 2)[0],
        "Suicidal": np.where(subject_group == 3)[0],
    }

    return group_indices
