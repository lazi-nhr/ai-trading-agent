"""
Custom training callbacks:
  - SharpeEvalCallback:  evaluates on Sharpe ratio (not mean reward) and saves the best model
  - ActionStatsCallback: monitors action diversity to detect policy collapse early
  - PolicyCollapseCallback: stops training on degenerate actions
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# Metrics helper (duplicated here to avoid circular import; keep lightweight)
# ---------------------------------------------------------------------------

def _compute_sharpe(equity: list[float], ann_factor: float) -> float:
    """Annualized Sharpe from an equity list."""
    if len(equity) < 3:
        return 0.0
    ec = pd.Series(equity)
    ret = ec.pct_change().dropna()
    mu = ret.mean() * ann_factor
    sigma = ret.std() * math.sqrt(ann_factor)
    return mu / (sigma + 1e-12)


def _scalar_bool(x) -> bool:
    arr = np.asarray(x)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _scalar_float(x) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _parse_step_result(result):
    """Parse gym/gymnasium step outputs for both Env and VecEnv."""
    if len(result) == 5:
        obs, reward, done, truncated, info = result
        done_flag = _scalar_bool(done) or _scalar_bool(truncated)
    else:
        obs, reward, done, info = result
        done_flag = _scalar_bool(done)

    if isinstance(info, (list, tuple)):
        info0 = info[0] if len(info) > 0 and isinstance(info[0], dict) else {}
    elif isinstance(info, dict):
        info0 = info
    else:
        info0 = {}

    return obs, _scalar_float(reward), done_flag, info0


# ============================================================================
# SharpeEvalCallback
# ============================================================================

class SharpeEvalCallback(BaseCallback):
    """
    Periodically runs full backtest episodes on a validation environment
    and saves the model that achieves the highest Sharpe ratio.

    Parameters
    ----------
    eval_env : gym.Env
        Validation environment (unwrapped or Monitor-wrapped).
    eval_freq : int
        Evaluate every ``eval_freq`` timesteps.
    n_eval_episodes : int
        Number of episodes per evaluation.
    best_model_save_path : str
        Directory to save the best model.
    ann_factor : float
        Annualization factor for Sharpe computation.
    log_path : str | None
        If set, write JSON lines of eval results.
    verbose : int
        Verbosity.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 14400,
        n_eval_episodes: int = 3,
        best_model_save_path: str = "./models",
        ann_factor: float = 365 * 24 * 60,
        log_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.ann_factor = ann_factor
        self.log_path = log_path

        self.best_sharpe = -np.inf
        self.eval_history: list[dict] = []

        os.makedirs(best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        sharpes = []
        mean_rewards = []
        final_values = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            equity = [1.0]
            ep_reward = 0.0
            done = False
            max_steps = 200_000

            for _ in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = _parse_step_result(self.eval_env.step(action))

                ep_reward += reward
                pv = info.get("portfolio_value", equity[-1])
                equity.append(pv)

                if done:
                    break

            sharpes.append(_compute_sharpe(equity, self.ann_factor))
            mean_rewards.append(ep_reward)
            final_values.append(equity[-1])

        avg_sharpe = float(np.mean(sharpes))
        avg_reward = float(np.mean(mean_rewards))
        avg_fv = float(np.mean(final_values))

        record = {
            "timestep": self.num_timesteps,
            "sharpe_mean": avg_sharpe,
            "reward_mean": avg_reward,
            "final_value_mean": avg_fv,
        }
        self.eval_history.append(record)

        improved = avg_sharpe > self.best_sharpe
        if improved:
            self.best_sharpe = avg_sharpe
            save_path = os.path.join(self.best_model_save_path, "best_sharpe_model")
            self.model.save(save_path)

        if self.verbose:
            tag = " ★ NEW BEST" if improved else ""
            msg = (
                f"\n{'='*60}\n"
                f"SHARPE EVAL @ step {self.num_timesteps:,}\n"
                f"  Avg Sharpe : {avg_sharpe:+.4f}{tag}\n"
                f"  Avg Reward : {avg_reward:+.4f}\n"
                f"  Avg FinalV : {avg_fv:.4f}\n"
                f"  Best Sharpe: {self.best_sharpe:+.4f}\n"
                f"{'='*60}\n"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        if self.log_path:
            import json
            os.makedirs(self.log_path, exist_ok=True)
            with open(os.path.join(self.log_path, "sharpe_eval.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")

        return True


# ============================================================================
# ActionStatsCallback
# ============================================================================

class ActionStatsCallback(BaseCallback):
    """Monitor action statistics during validation to detect collapse."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 14400,
        n_eval_episodes: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.action_history: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        all_actions = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            for _ in range(10_000):
                action, _ = self.model.predict(obs, deterministic=True)
                val = float(action.flat[0]) if isinstance(action, np.ndarray) else float(action)
                all_actions.append(val)
                obs, _, done, _ = _parse_step_result(self.eval_env.step(action))
                if done:
                    break

        if not all_actions:
            return True

        arr = np.array(all_actions)
        rec = {
            "step": self.num_timesteps,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        self.action_history.append(rec)

        lines = [
            f"\nACTION STATS @ {self.num_timesteps:,}: "
            f"mean={rec['mean']:.4f} std={rec['std']:.4f} "
            f"[{rec['min']:.4f}, {rec['max']:.4f}]",
        ]
        if rec["std"] < 0.01:
            lines.append("  ⚠️  VERY LOW diversity — possible collapse!")
        elif rec["std"] < 0.05:
            lines.append("  ⚠️  Low diversity")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        return True


# ============================================================================
# PolicyCollapseCallback
# ============================================================================

class PolicyCollapseCallback(BaseCallback):
    """Stop training if actions become degenerate."""

    def __init__(
        self,
        eval_env,
        check_freq: int = 5000,
        action_std_threshold: float = 0.01,
        action_extreme_threshold: float = 0.99,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.action_std_threshold = action_std_threshold
        self.action_extreme_threshold = action_extreme_threshold

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        obs = self.eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        actions = []
        for _ in range(100):
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(float(action.flat[0]) if isinstance(action, np.ndarray) else float(action))
            obs, _, done, _ = _parse_step_result(self.eval_env.step(action))
            if done:
                obs = self.eval_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        arr = np.array(actions)
        std = arr.std()
        mean = arr.mean()

        collapse = std < self.action_std_threshold or abs(mean) > self.action_extreme_threshold

        if collapse:
            msg = (
                f"\n⚠️  POLICY COLLAPSE @ step {self.num_timesteps:,}  "
                f"mean={mean:.4f} std={std:.4f}\n"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()
            return False  # stop training

        return True
