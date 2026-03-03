from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 1000
    test_size: int = 250
    min_train: int = 500
    proba_threshold: float = 0.55


def walk_forward_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: WalkForwardConfig,
    trainer: Callable[[pd.DataFrame, pd.Series], Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
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

        model = trainer(X_train, y_train)

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

    metrics: Dict[str, Any] = {}
    if len(y_true_all) > 0:
        yt = np.array(y_true_all)
        yp = np.array(y_pred_all)
        pr = np.array(proba_all)
        metrics["accuracy"] = float(accuracy_score(yt, yp))
        metrics["f1"] = float(f1_score(yt, yp))
        metrics["roc_auc"] = float(roc_auc_score(yt, pr)) if len(np.unique(yt)) == 2 else None
        metrics["n_pred"] = int(len(yt))
    else:
        metrics["accuracy"] = None
        metrics["f1"] = None
        metrics["roc_auc"] = None
        metrics["n_pred"] = 0

    return proba, metrics