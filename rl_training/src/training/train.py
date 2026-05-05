"""
train.py — Training pipeline utilities.

Provides functions to:
  - create_model: build a PPO model with CNN feature extractor + VecNormalize
  - train_single: train one model for a given (fee, frequency) config
  - train_multi_fee: sweep over fee structures
  - train_multi_frequency: sweep over frequencies (and optionally fees)
"""

from __future__ import annotations

import gc
import os
import sys
from copy import deepcopy

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..envs.stat_arb_env import StatArbEnv
from ..models.feature_extractors import CNNFeatureExtractor
from .callbacks import SharpeEvalCallback, ActionStatsCallback, PolicyCollapseCallback

# Re-export ANNUALIZATION from config (imported by callers)
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
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    X: np.ndarray,
    R: np.ndarray,
    VOL: np.ndarray,
    tickers: list[str],
    lookback: int,
    cfg_env: dict,
    frequency: str = "1m",
) -> StatArbEnv:
    """Create a Monitor-wrapped StatArbEnv."""
    env = StatArbEnv(X, R, VOL, tickers, lookback, cfg_env, frequency=frequency)
    return Monitor(env, filename=None)


def make_vec_env(
    X: np.ndarray,
    R: np.ndarray,
    VOL: np.ndarray,
    tickers: list[str],
    lookback: int,
    cfg_env: dict,
    cfg_rl: dict,
    frequency: str = "1m",
) -> VecNormalize | DummyVecEnv:
    """Create a DummyVecEnv, optionally wrapped with VecNormalize."""
    vec = DummyVecEnv([lambda: make_env(X, R, VOL, tickers, lookback, cfg_env, frequency)])

    if cfg_rl.get("normalize_obs", False) or cfg_rl.get("normalize_reward", False):
        vec = VecNormalize(
            vec,
            norm_obs=cfg_rl.get("normalize_obs", True),
            norm_reward=cfg_rl.get("normalize_reward", True),
            clip_obs=cfg_rl.get("clip_obs", 10.0),
            gamma=cfg_rl.get("gamma", 0.99),
        )
    return vec


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    vec_env,
    cfg_rl: dict,
    cfg_env: dict,
    n_features: int,
    lookback: int,
    n_pairs: int,
    tb_log: str | None = None,
    device: str = "cpu",
) -> PPO:
    """
    Create a PPO model with optional CNN feature extractor.

    Parameters
    ----------
    vec_env : VecEnv
        Vectorised training environment.
    cfg_rl : dict
        RL hyperparameters from CONFIG["RL"].
    cfg_env : dict
        Environment config from CONFIG["ENV"] (needed for n_pairs etc.).
    n_features, lookback, n_pairs : int
        Dimensions for the CNN feature extractor.
    """

    # ----- Policy kwargs -----
    policy_kwargs = {}

    if cfg_rl.get("feature_extractor", "mlp") == "cnn":
        policy_kwargs["features_extractor_class"] = CNNFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(
            n_features=n_features,
            lookback=lookback,
            n_pairs=n_pairs,
            features_dim=cfg_rl.get("cnn_features_dim", 128),
        )

    pi_arch = cfg_rl.get("net_arch_pi", [64, 64])
    vf_arch = cfg_rl.get("net_arch_vf", [64, 64])
    policy_kwargs["net_arch"] = dict(pi=pi_arch, vf=vf_arch)

    if cfg_rl.get("use_sde", False):
        policy_kwargs["log_std_init"] = -2.0

    # ----- Build model -----
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LinearSchedule(
            start=cfg_rl.get("learning_rate", 3e-4),
            end=1e-5,
            end_fraction=1.0,
        ),
        n_steps=cfg_rl.get("n_steps", 2048),
        batch_size=cfg_rl.get("batch_size", 128),
        n_epochs=cfg_rl.get("n_epochs", 10),
        gamma=cfg_rl.get("gamma", 0.995),
        gae_lambda=cfg_rl.get("gae_lambda", 0.95),
        clip_range=cfg_rl.get("clip_range", 0.2),
        clip_range_vf=cfg_rl.get("clip_range_vf", None),
        ent_coef=cfg_rl.get("ent_coef", 0.05),
        vf_coef=cfg_rl.get("vf_coef", 0.5),
        max_grad_norm=cfg_rl.get("max_grad_norm", 0.5),
        use_sde=cfg_rl.get("use_sde", False),
        sde_sample_freq=cfg_rl.get("sde_sample_freq", -1),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log,
        device=device,
        verbose=0,
    )
    return model


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single(
    X_train: np.ndarray,
    R_train: np.ndarray,
    VOL_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    VOL_val: np.ndarray,
    tickers: list[str],
    lookback: int,
    cfg: dict,
    frequency: str = "1m",
    fee_bps: float | None = None,
    models_dir: str = "./models",
    tag: str = "",
    verbose: bool = True,
) -> str:
    """
    Train a single PPO model and return the save path of the best model.

    If ``fee_bps`` is given, it overrides ``cfg["ENV"]["transaction_costs"]["taker_bps"]``.
    """
    cfg = deepcopy(cfg)
    if fee_bps is not None:
        cfg["ENV"]["transaction_costs"]["taker_bps"] = fee_bps

    cfg_env = cfg["ENV"]
    cfg_rl = cfg["RL"]
    n_features = X_train.shape[2]
    n_pairs = X_train.shape[1]

    os.makedirs(models_dir, exist_ok=True)

    # Build environments
    vec_train = make_vec_env(X_train, R_train, VOL_train, tickers, lookback, cfg_env, cfg_rl, frequency)
    vec_val = make_vec_env(X_val, R_val, VOL_val, tickers, lookback, cfg_env, cfg_rl, frequency)

    model = create_model(
        vec_train,
        cfg_rl,
        cfg_env,
        n_features=n_features,
        lookback=lookback,
        n_pairs=n_pairs,
        tb_log=cfg.get("IO", {}).get("tb_logdir"),
    )

    ann_factor = ANNUALIZATION.get(frequency, 365 * 24 * 60)

    # Callbacks
    sharpe_cb = SharpeEvalCallback(
        eval_env=vec_val,
        eval_freq=cfg.get("EVAL", {}).get("frequency", 14400),
        n_eval_episodes=cfg.get("EVAL", {}).get("n_eval_episodes", 3),
        best_model_save_path=models_dir,
        ann_factor=ann_factor,
        log_path=models_dir,
        verbose=1 if verbose else 0,
    )

    action_cb = ActionStatsCallback(
        eval_env=vec_val,
        eval_freq=cfg.get("EVAL", {}).get("frequency", 14400),
        n_eval_episodes=1,
    )

    collapse_cb = PolicyCollapseCallback(
        eval_env=vec_val,
        check_freq=cfg.get("EVAL", {}).get("frequency", 14400),
        action_std_threshold=cfg.get("EVAL", {}).get("action_std_threshold", 0.01),
        action_extreme_threshold=cfg.get("EVAL", {}).get("action_extreme_threshold", 0.99),
    )

    callback = CallbackList([sharpe_cb, action_cb, collapse_cb])

    timesteps = int(cfg_rl.get("timesteps", 2e6))

    label = tag or f"freq{frequency}_fee{fee_bps}"
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING: {label}")
        print(f"  frequency={frequency}  fee={fee_bps} bps  timesteps={timesteps:,}")
        print(f"  reward={cfg_env['reward']['type']}  extractor={cfg_rl.get('feature_extractor','mlp')}")
        print(f"{'='*60}")

    try:
        # Rich progress bars can recurse/crash in some notebook renderers.
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    except KeyboardInterrupt:
        sys.stdout.write(f"\n⚠️  Interrupted: {label}\n")
        sys.stdout.flush()
    finally:
        gc.collect()

    # Save final model
    final_path = os.path.join(models_dir, f"{label}_final.zip")
    model.save(final_path)

    # Also save VecNormalize stats if used
    if isinstance(vec_train, VecNormalize):
        norm_path = os.path.join(models_dir, f"{label}_vecnorm.pkl")
        vec_train.save(norm_path)

    best_path = os.path.join(models_dir, "best_sharpe_model.zip")
    if os.path.exists(best_path):
        if verbose:
            print(f"Best Sharpe model: {best_path}  (Sharpe={sharpe_cb.best_sharpe:.4f})")
    if verbose:
        print(f"Final model: {final_path}")

    del model
    gc.collect()
    return best_path if os.path.exists(best_path) else final_path


# ---------------------------------------------------------------------------
# Multi-fee sweep
# ---------------------------------------------------------------------------

def train_multi_fee(
    fee_structures: list[float],
    X_train: np.ndarray,
    R_train: np.ndarray,
    VOL_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    VOL_val: np.ndarray,
    tickers: list[str],
    lookback: int,
    cfg: dict,
    frequency: str = "1m",
    models_dir: str = "./models",
    verbose: bool = True,
) -> dict[float, str]:
    """Train one model per fee level. Returns {fee_bps: model_path}."""
    results = {}
    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-FEE TRAINING  |  freq={frequency}  |  fees={fee_structures}")
        print(f"{'='*70}")

    for fee in fee_structures:
        tag = f"freq{frequency}_fee{fee}bps"
        sub_dir = os.path.join(models_dir, tag)
        path = train_single(
            X_train, R_train, VOL_train,
            X_val, R_val, VOL_val,
            tickers, lookback, cfg,
            frequency=frequency,
            fee_bps=fee,
            models_dir=sub_dir,
            tag=tag,
            verbose=verbose,
        )
        results[fee] = path

    if verbose:
        print(f"\n{'='*70}")
        print("MULTI-FEE COMPLETE")
        for fee, p in results.items():
            print(f"  {fee:6.2f} bps → {p}")
        print(f"{'='*70}")
    return results


# ---------------------------------------------------------------------------
# Multi-frequency sweep
# ---------------------------------------------------------------------------

def train_multi_frequency(
    frequencies: list[str],
    fee_structures: list[float],
    data_by_freq: dict,  # freq → (X_train, R_train, VOL_train, X_val, R_val, VOL_val)
    tickers: list[str],
    lookback: int,
    cfg: dict,
    models_dir: str = "./models",
    verbose: bool = True,
) -> dict[str, dict[float, str]]:
    """
    Train models for every (frequency, fee) combination.

    Parameters
    ----------
    frequencies : list[str]
        e.g. ["1m", "5m", "30m"]
    fee_structures : list[float]
        e.g. [0.0, 1.7, 5.0]
    data_by_freq : dict
        Mapping from frequency string to a tuple of
        (X_train, R_train, VOL_train, X_val, R_val, VOL_val).
    """
    all_results: dict[str, dict[float, str]] = {}

    if verbose:
        total = len(frequencies) * len(fee_structures)
        print(f"\n{'='*70}")
        print("MULTI-FREQUENCY x MULTI-FEE TRAINING")
        print(f"  Frequencies: {frequencies}")
        print(f"  Fee structures: {fee_structures}")
        print(f"  Total models: {total}")
        print(f"{'='*70}")

    for freq in frequencies:
        if freq not in data_by_freq:
            print(f"⚠️  No data for frequency {freq}, skipping.")
            continue

        X_tr, R_tr, V_tr, X_va, R_va, V_va = data_by_freq[freq]
        freq_dir = os.path.join(models_dir, f"freq_{freq}")

        results = train_multi_fee(
            fee_structures=fee_structures,
            X_train=X_tr,
            R_train=R_tr,
            VOL_train=V_tr,
            X_val=X_va,
            R_val=R_va,
            VOL_val=V_va,
            tickers=tickers,
            lookback=lookback,
            cfg=cfg,
            frequency=freq,
            models_dir=freq_dir,
            verbose=verbose,
        )
        all_results[freq] = results

    if verbose:
        print(f"\n{'='*70}")
        print("ALL TRAINING COMPLETE")
        for freq, fee_dict in all_results.items():
            for fee, path in fee_dict.items():
                print(f"  [{freq}] {fee:.2f} bps → {path}")
        print(f"{'='*70}")

    return all_results
