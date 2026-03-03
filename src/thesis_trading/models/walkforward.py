from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from thesis_trading.models.logreg import train_logistic


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 1000   # bars
    test_size: int = 250     # bars
    min_train: int = 500     # safety
    proba_threshold: float = 0.55  # for long; short uses (1 - p)


def walk_forward_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: WalkForwardConfig,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Walk-forward training:
    - Train on a rolling window (train_size)
    - Predict proba on next window (test_size)
    Returns:
      proba_up: probability of class 1 (up) aligned with X index
      metrics: aggregated classification metrics on predicted windows
    """
    n = len(X)
    proba = pd.Series(np.nan, index=X.index, name="proba_up")

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    proba_all: List[float] = []

    start = 0
    while True:
        train_start = start
        train_end = train_start + cfg.train_size
        test_end = train_end + cfg.test_size

        if train_end >= n:
            break
        if test_end > n:
            test_end = n

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        if len(X_train) < cfg.min_train:
            start += cfg.test_size
            continue

        model = train_logistic(X_train, y_train)

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        p = model.predict_proba(X_test)[:, 1]
        proba.iloc[train_end:test_end] = p

        y_hat = (p >= 0.5).astype(int)

        y_true_all.extend(y_test.astype(int).tolist())
        y_pred_all.extend(y_hat.tolist())
        proba_all.extend(p.tolist())

        if test_end == n:
            break

        start += cfg.test_size

    # metrics
    metrics: Dict[str, Any] = {}
    if len(y_true_all) > 0:
        yt = np.array(y_true_all)
        yp = np.array(y_pred_all)
        pr = np.array(proba_all)
        metrics["accuracy"] = float(accuracy_score(yt, yp))
        metrics["f1"] = float(f1_score(yt, yp))
        # roc_auc needs both classes present
        if len(np.unique(yt)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(yt, pr))
        else:
            metrics["roc_auc"] = None
        metrics["n_pred"] = int(len(yt))
    else:
        metrics["accuracy"] = None
        metrics["f1"] = None
        metrics["roc_auc"] = None
        metrics["n_pred"] = 0

    return proba, metrics