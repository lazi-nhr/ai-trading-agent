"""
backtest.py — Backtesting, metrics computation, and plotting utilities.
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import ANNUALIZATION
except ImportError:
    ANNUALIZATION = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "1d": 365,
    }


# ---------------------------------------------------------------------------
# Annualization
# ---------------------------------------------------------------------------

def annualize_factor(sampling: str) -> float:
    return ANNUALIZATION.get(sampling, 365 * 24)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_curve: pd.Series,
    sampling: str,
    turnover_series: Optional[pd.Series] = None,
) -> dict:
    """Compute standard portfolio performance metrics."""
    ret = equity_curve.pct_change().dropna()
    ann = annualize_factor(sampling)
    mu = ret.mean() * ann
    sigma = ret.std() * math.sqrt(ann)
    sharpe = mu / (sigma + 1e-12)

    downside = ret[ret < 0].std() * math.sqrt(ann)
    sortino = mu / (downside + 1e-12)

    if len(equity_curve) > 1:
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            dt = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / (365 * 24 * 3600)
        else:
            dt = len(equity_curve) / ann
        dt = max(float(dt), 1e-12)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / dt) - 1
    else:
        cagr = 0.0

    cummax = equity_curve.cummax()
    maxdd = float((equity_curve / cummax - 1).min())
    calmar = mu / (abs(maxdd) + 1e-12)
    hit = float((ret > 0).mean())
    turnover = float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) > 0 else np.nan

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": maxdd,
        "Calmar": calmar,
        "Volatility": sigma,
        "Turnover": turnover,
        "HitRatio": hit,
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def backtest_env(
    env,
    model=None,
    include_leverage: bool = False,
    include_returns: bool = False,
):
    """
    Run a full episode through ``env`` using ``model`` (or cash if None).

    Returns (equity_curve, turnover, [leverage, actions, returns]).
    """
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    obs, _ = env.reset()

    pv, turns, levs, actions, rets = [], [], [], [], []

    total_steps = len(unwrapped.R) - 1

    for _ in range(total_steps):
        if model is None:
            action = np.array([0.0])
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, done, _, info = env.step(action)
        pv.append(info["portfolio_value"])
        turns.append(info["turnover"])
        levs.append(info.get("total_leverage", 0.0))
        actions.append(info.get("action_taken", action.flat[0] if isinstance(action, np.ndarray) else action))
        rets.append(info.get("portfolio_log_ret", 0.0))

        if done:
            break

    idx = pd.RangeIndex(len(pv))
    ec = pd.Series(pv, index=idx)
    to = pd.Series(turns, index=idx)
    lev = pd.Series(levs, index=idx)
    act = pd.Series(actions, index=idx)
    ret = pd.Series(rets, index=idx)

    if include_returns:
        return ec, to, lev, act, ret
    if include_leverage:
        return ec, to, lev, act
    return ec, to


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity_curves(
    curves: dict[str, pd.Series],
    title: str = "Equity Curves",
    sampling: str = "1m",
    save_path: str | None = None,
):
    """
    Plot named equity curves on one axis.

    Parameters
    ----------
    curves : dict[str, pd.Series]
        Mapping of label → equity curve Series.
    """
    bars_per_day = annualize_factor(sampling) / 365
    fig, ax = plt.subplots(figsize=(16, 6))

    for label, ec in curves.items():
        days = np.arange(len(ec)) / bars_per_day
        ax.plot(days, ec.values, label=label, linewidth=2)

    ax.axhline(1.0, color="gray", ls=":", alpha=0.4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


def plot_comparison(
    all_results: list[dict],
    baseline_result: dict | None = None,
    sampling: str = "1m",
    save_dir: str | None = None,
):
    """
    Multi-panel comparison: equity, drawdown, rolling actions.

    Each entry in ``all_results`` should have keys:
        model_name, equity_curve (pd.Series), actions (pd.Series), color, linestyle
    """
    bars_per_day = annualize_factor(sampling) / 365

    # --- equity ---
    fig1, ax = plt.subplots(figsize=(16, 6))
    for r in all_results:
        d = np.arange(len(r["equity_curve"])) / bars_per_day
        ax.plot(d, r["equity_curve"].values, label=r["model_name"], color=r.get("color"), ls=r.get("linestyle", "-"), lw=2)
    if baseline_result:
        d = np.arange(len(baseline_result["equity_curve"])) / bars_per_day
        ax.plot(d, baseline_result["equity_curve"].values, label=baseline_result["model_name"], color="gray", ls="--", lw=2)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.4)
    ax.set_title("Portfolio Value", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_dir:
        fig1.savefig(f"{save_dir}/equity_comparison.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig1)

    # --- drawdown ---
    fig2, ax = plt.subplots(figsize=(16, 5))
    for r in all_results:
        ec = r["equity_curve"]
        dd = (ec / ec.cummax() - 1) * 100
        d = np.arange(len(dd)) / bars_per_day
        ax.plot(d, dd.values, label=r["model_name"], color=r.get("color"), lw=2)
    if baseline_result:
        ec = baseline_result["equity_curve"]
        dd = (ec / ec.cummax() - 1) * 100
        d = np.arange(len(dd)) / bars_per_day
        ax.plot(d, dd.values, label=baseline_result["model_name"], color="gray", ls="--", lw=2)
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_title("Drawdown (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_dir:
        fig2.savefig(f"{save_dir}/drawdown_comparison.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig2)

    # --- actions ---
    fig3, ax = plt.subplots(figsize=(16, 5))
    roll_win = max(1, int(bars_per_day))  # 1-day rolling mean
    for r in all_results:
        act = r.get("actions")
        if act is None:
            continue
        rm = act.rolling(window=roll_win, min_periods=1).mean()
        d = np.arange(len(rm)) / bars_per_day
        ax.plot(d, rm.values, label=r["model_name"], color=r.get("color"), lw=2)
    ax.axhline(0, color="black", lw=1.5, alpha=0.5, label="Neutral")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Trading Actions (1-day rolling mean)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Action")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_dir:
        fig3.savefig(f"{save_dir}/actions_comparison.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig3)
