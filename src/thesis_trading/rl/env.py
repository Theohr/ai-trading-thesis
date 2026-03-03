from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class TradingEnvConfig:
    cost_bps: float = 1.5       # per trade change (in bps)
    slippage_bps: float = 0.0   # optional extra cost
    max_episode_steps: Optional[int] = None  # None = full dataset
    reward_scale: float = 1.0   # scale rewards if needed


class ForexTradingEnv(gym.Env):
    """
    Minimal discrete-action trading environment.

    Actions: 0=SHORT, 1=FLAT, 2=LONG
    State: feature vector at time t
    Reward: position_t * return_{t->t+1} - cost * |position_t - position_{t-1}|
      - This approximates trading at close t and holding to close t+1.

    Notes:
    - Uses Close returns only (consistent with your project).
    - No leakage: observation at t does not include info from t+1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        cfg: TradingEnvConfig,
        seed: int = 42,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Precompute next-step returns r[t] = Close[t+1]/Close[t]-1
        close = self.df["Close"].astype(float).values
        rets = (close[1:] / close[:-1]) - 1.0
        self._next_ret = np.concatenate([rets, np.array([0.0], dtype=float)])  # last unused

        # Spaces
        n_features = len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Internal state
        self._t = 0
        self._pos_prev = 0  # -1, 0, +1
        self._steps = 0

    def _action_to_pos(self, action: int) -> int:
        # 0 short, 1 flat, 2 long
        return -1 if action == 0 else (0 if action == 1 else 1)

    def _get_obs(self) -> np.ndarray:
        x = self.df.loc[self._t, self.feature_cols].to_numpy(dtype=np.float32)
        return x

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._t = 0
        self._pos_prev = 0
        self._steps = 0

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        pos = self._action_to_pos(int(action))

        # Transaction cost for changing position
        turnover = abs(pos - self._pos_prev)
        cost = (self.cfg.cost_bps + self.cfg.slippage_bps) * 1e-4 * turnover

        # Next return from t -> t+1
        r = float(self._next_ret[self._t])

        # Reward is PnL minus cost
        reward = (pos * r) - cost
        reward *= float(self.cfg.reward_scale)

        # Move forward
        self._pos_prev = pos
        self._t += 1
        self._steps += 1

        terminated = False
        truncated = False

        # End of data (can't compute further meaningful reward)
        if self._t >= len(self.df) - 1:
            terminated = True

        # Optional episode limit
        if self.cfg.max_episode_steps is not None and self._steps >= self.cfg.max_episode_steps:
            truncated = True

        obs = self._get_obs() if not (terminated or truncated) else np.zeros(
            (len(self.feature_cols),), dtype=np.float32
        )

        info = {
            "t": self._t,
            "pos": pos,
            "ret": r,
            "cost": cost,
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info