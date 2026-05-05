"""
StatArbEnv — Improved RL environment for statistical arbitrage.

Key improvements over the original PortfolioWeightsEnvUtility:
1. Differential Sharpe Ratio reward (Moody & Saffell 1998)
2. Fixed-length episodes with uniform random start sampling
3. Pair identity in observations (one-hot encoded)
4. Vol-dependent slippage
5. Observation structured as (n_features, lookback) — not flattened
   (flattening is deferred to the feature extractor for CNN compatibility)
6. Reward scaling so value function can learn effectively
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StatArbEnv(gym.Env):
    """
    Statistical Arbitrage environment with continuous action space.

    Observation layout (flat vector):
        [ market_features(n_features * lookback) | position(3) | pair_id(n_pairs) ]

    Action:
        Scalar in [-1, 1].
        -1 → max short asset1 / long asset2
         0 → 100 % cash
        +1 → max long asset1 / short asset2

    Reward (selectable via cfg_env["reward"]["type"]):
        "sharpe"  → differential Sharpe ratio  (default)
        "utility" → quadratic utility  x - (λ/2)x²
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        X: np.ndarray,
        R: np.ndarray,
        VOL: np.ndarray,
        tickers: list[str],
        lookback: int,
        cfg_env: dict,
        frequency: str = "1m",
    ):
        super().__init__()

        # Core data  (samples, n_pairs, n_features, lookback)
        self.X = X
        self.R = R                 # (samples, 2)  [asset1_ret, asset2_ret]
        self.VOL = VOL             # (samples, 1)
        self.tickers = tickers
        self.lookback = lookback
        self.cfg = cfg_env
        self.frequency = frequency

        self.n_pairs = X.shape[1]
        self.n_features = X.shape[2]
        self.n_assets = 2
        self.include_cash = cfg_env.get("include_cash", True)

        # ----- Episode length -----
        ep_cfg = cfg_env.get("episode_length", None)
        if ep_cfg is not None:
            self.episode_length = int(ep_cfg)
        else:
            ep_map = cfg_env.get("episode_length_map", {})
            self.episode_length = int(ep_map.get(frequency, 2016))

        # ----- Observation space -----
        market_obs_dim = self.n_features * lookback
        position_obs_dim = 3                    # [w_asset1, w_asset2, w_cash]
        pair_id_dim = self.n_pairs              # one-hot pair identity
        obs_dim = market_obs_dim + position_obs_dim + pair_id_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ----- Action space -----
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ----- Transaction costs -----
        tc = cfg_env.get("transaction_costs", {})
        self.taker_fee = tc.get("taker_bps", 0.0) / 1e4
        self.base_slippage = tc.get("base_slippage_bps", tc.get("slippage_bps", 0.0)) / 1e4
        self.vol_slippage_mult = tc.get("vol_slippage_multiplier", 0.0)

        # ----- Reward -----
        rew_cfg = cfg_env.get("reward", {})
        self.reward_type = rew_cfg.get("type", "sharpe")
        self.reward_clip = rew_cfg.get("reward_clip", 3.0)
        self.reward_scale = rew_cfg.get("reward_scale", 10000)

        # Differential Sharpe state
        self.sharpe_eta = rew_cfg.get("sharpe_eta", 0.01)
        self._A = 0.0   # running mean of returns
        self._B = 0.0   # running mean of squared returns

        # Quadratic utility fallback
        self.lambda_utility = rew_cfg.get("lambda_utility", 6.0)
        self.lambda_basic = rew_cfg.get("lambda_basic", 0.01)

        # ----- Internal state -----
        self.active_pair_idx = 0
        self.t = 0
        self.t_start = 0
        self.portfolio_value = 1.0
        self.w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.last_action = 0.0

        self.reset(seed=cfg_env.get("seed", 42))

    # ------------------------------------------------------------------
    # Action → weights
    # ------------------------------------------------------------------
    def _continuous_to_weights(self, action: float) -> np.ndarray:
        action = float(np.clip(action, -1.0, 1.0))
        pos = action * 0.5
        return np.array([pos, -pos, 1.0 - 2 * abs(pos)], dtype=np.float64)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _to_obs(self, t: int) -> np.ndarray:
        # Market features: (n_features, lookback) → flat
        market = self.X[t, self.active_pair_idx, :, :].reshape(-1).astype(np.float32)
        # Guard against non-finite tensor values (NaN/Inf) from upstream data.
        market = np.nan_to_num(market, nan=0.0, posinf=1e6, neginf=-1e6)
        # Portfolio weights
        position = np.nan_to_num(self.w.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        # Pair identity (one-hot)
        pair_id = np.zeros(self.n_pairs, dtype=np.float32)
        pair_id[self.active_pair_idx] = 1.0
        obs = np.concatenate([market, position, pair_id])
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _diff_sharpe_reward(self, net_return: float) -> float:
        """Differential Sharpe ratio (Moody & Saffell 1998)."""
        eta = self.sharpe_eta
        delta_A = net_return - self._A
        delta_B = net_return ** 2 - self._B

        denom = (self._B - self._A ** 2) ** 1.5
        if abs(denom) < 1e-12:
            # Not enough history; use simple return as reward
            reward = net_return * self.reward_scale
        else:
            reward = (self._B * delta_A - 0.5 * self._A * delta_B) / denom

        # Update running statistics
        self._A += eta * delta_A
        self._B += eta * delta_B
        return reward

    def _utility_reward(self, net_return: float) -> float:
        """Quadratic utility: x - (λ/2) x²."""
        x = net_return * self.reward_scale
        return x - (self.lambda_utility / 2.0) * (x ** 2)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Pick a random pair
        self.active_pair_idx = int(self.np_random.integers(0, self.n_pairs))

        # Pick a random start so that the full episode fits
        max_start = len(self.R) - self.episode_length - 1
        if max_start < 1:
            max_start = 1
        self.t = int(self.np_random.integers(0, max_start))
        self.t_start = self.t

        # Reset portfolio
        self.portfolio_value = 1.0
        self.w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.last_action = 0.0

        # Reset differential Sharpe state
        self._A = 0.0
        self._B = 0.0

        return self._to_obs(self.t), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action.flat[0])
        else:
            action = float(action)

        # Target weights
        w_target = self._continuous_to_weights(action)
        turnover = float(np.sum(np.abs(w_target[:2] - self.w[:2])))

        # Vol-dependent slippage
        pair_vol = float(self.VOL[self.t, 0]) if self.VOL.shape[1] == 1 else float(self.VOL[self.t, self.active_pair_idx])
        if not np.isfinite(pair_vol):
            pair_vol = 0.0
        slippage = self.base_slippage * (1.0 + pair_vol * self.vol_slippage_mult)
        trading_cost = (self.taker_fee + slippage) * turnover

        # Update weights
        self.w = w_target

        # Asset returns (clipped for numerical safety)
        r1 = float(np.clip(self.R[self.t, 0], -0.1, 0.1))
        r2 = float(np.clip(self.R[self.t, 1], -0.1, 0.1))
        if not np.isfinite(r1):
            r1 = 0.0
        if not np.isfinite(r2):
            r2 = 0.0

        portfolio_log_ret = self.w[0] * r1 + self.w[1] * r2
        net_return = portfolio_log_ret - trading_cost

        # ----- Reward -----
        if self.reward_type == "sharpe":
            reward = self._diff_sharpe_reward(net_return)
        elif self.reward_type == "utility":
            reward = self._utility_reward(net_return)
        else:
            # basic: return - lambda * vol
            inst_vol = pair_vol * (abs(self.w[0]) + abs(self.w[1]))
            reward = (net_return * self.reward_scale) - self.lambda_basic * inst_vol

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        if not np.isfinite(reward):
            reward = 0.0

        # Update portfolio value
        if not np.isfinite(net_return):
            net_return = 0.0
        self.portfolio_value *= math.exp(net_return)
        if not np.isfinite(self.portfolio_value):
            self.portfolio_value = 1.0
        self.last_action = action

        # Advance
        self.t += 1
        steps_done = self.t - self.t_start
        terminated = (steps_done >= self.episode_length) or (self.t >= len(self.R) - 1)
        truncated = False

        obs = self._to_obs(self.t) if not terminated else self._to_obs(self.t - 1)

        info = {
            "portfolio_value": self.portfolio_value,
            "total_leverage": float(np.sum(np.abs(self.w[:2]))),
            "turnover": turnover,
            "portfolio_log_ret": portfolio_log_ret,
            "net_return": net_return,
            "reward": reward,
            "action_taken": action,
            "pair_idx": self.active_pair_idx,
            "slippage_effective": slippage,
        }
        return obs, reward, terminated, truncated, info
