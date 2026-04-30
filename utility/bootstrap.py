import numpy as np


def uniform_bootstrap_trials(X, trial_indices, n_aug=200, k=12, replace=True):
    """
    X: ch x time x trial

    Output:
        X_aug: n_aug x ch x time
    """

    trial_indices = np.array(trial_indices, dtype=int)

    if len(trial_indices) == 0:
        return None

    X_aug = []

    for _ in range(n_aug):
        sampled = np.random.choice(
            trial_indices,
            size=k,
            replace=replace,
        )

        avg_trial = X[:, :, sampled].mean(axis=2)
        X_aug.append(avg_trial)

    return np.stack(X_aug, axis=0)


def congruence_contrast_bootstrap(
    X,
    congruent_idx,
    incongruent_idx,
    n_aug=200,
    k=12,
    replace=True,
):
    """
    Feature = bootstrap_average(congruent) - bootstrap_average(incongruent)

    X: ch x time x trial

    Output:
        X_aug: n_aug x ch x time
    """

    congruent_idx = np.array(congruent_idx, dtype=int)
    incongruent_idx = np.array(incongruent_idx, dtype=int)

    if len(congruent_idx) == 0 or len(incongruent_idx) == 0:
        return None

    X_aug = []

    for _ in range(n_aug):
        con_sampled = np.random.choice(
            congruent_idx,
            size=k,
            replace=replace,
        )

        incon_sampled = np.random.choice(
            incongruent_idx,
            size=k,
            replace=replace,
        )

        con_avg = X[:, :, con_sampled].mean(axis=2)
        incon_avg = X[:, :, incon_sampled].mean(axis=2)

        contrast = con_avg - incon_avg
        X_aug.append(contrast)

    return np.stack(X_aug, axis=0)
