from __future__ import annotations

from io import BytesIO
from urllib.request import urlopen
import zipfile

import numpy as np


def load_banknote_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load the UCI Banknote Authentication dataset."""

    url = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
    with urlopen(url) as response:
        raw = response.read()
    with zipfile.ZipFile(BytesIO(raw)) as zf:
        with zf.open("data_banknote_authentication.txt") as f:
            data = np.loadtxt(f, delimiter=",", dtype=np.float32)
    x = data[:, :4]
    y = data[:, 4].astype(np.int64)
    return x, y


def make_pu_task(
    x: np.ndarray,
    y: np.ndarray,
    *,
    positive_label: int = 0,
    labeled_positive_size: int = 64,
    unlabeled_positive_size: int = 200,
    unlabeled_outlier_size: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a binary dataset into a simple PU task used in the README example."""

    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == positive_label)[0]
    neg_idx = np.where(y != positive_label)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    labeled_idx = pos_idx[:labeled_positive_size]
    unlabeled_pos_idx = pos_idx[
        labeled_positive_size : labeled_positive_size + unlabeled_positive_size
    ]
    unlabeled_neg_idx = neg_idx[:unlabeled_outlier_size]

    x_labeled = x[labeled_idx]
    x_unlabeled = np.concatenate([x[unlabeled_pos_idx], x[unlabeled_neg_idx]], axis=0)
    y_unlabeled_true = np.concatenate(
        [
            np.zeros(len(unlabeled_pos_idx), dtype=np.int64),
            np.ones(len(unlabeled_neg_idx), dtype=np.int64),
        ],
        axis=0,
    )

    perm = rng.permutation(len(y_unlabeled_true))
    x_unlabeled = x_unlabeled[perm]
    y_unlabeled_true = y_unlabeled_true[perm]
    return x_labeled, x_unlabeled, y_unlabeled_true
