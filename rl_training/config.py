CONFIG = {
    "DATA": {
        "forward_fill": True,
        "drop_na_after_ffill": True,
        "cache_data": True,
        "cache_dir": "./data_cache",
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "asset_price_format": "{ASSET}_{FEATURE}",
        "pair_feature_format": "{ASSET1}_{ASSET2}_{FEATURE}",
        "timestamp_col": "timestamp",
        "sampling": "1m",  # Base sampling rate of the raw data
        "features": {
            "file_id": "1OCqEkOWV73Z8e-67fpqVL3r3ugVcfml8",
            "file_name": "bin_futures_full_features",
            "type": "csv",
            "separator": ",",
            "index": "datetime",
            "start": "2024-05-01 00:00:00",
            "end": "2025-05-01 00:00:00",
            "individual_identifier": "close",
            "pair_identifier": "beta",
        },
    },

    "ENV": {
        "include_cash": True,
        "beta_adjusted_portfolio": False,

        # ----- Frequency Configuration -----
        # Base frequency of raw data; experiments resample from this
        "frequency": ["1m", "5m", "30m"],  # Frequencies for multi-frequency experiment

        "action_space_type": "continuous",
        "discrete_actions": [-1.0, -0.9, -0.8, -0.7, -0.6,
                             -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                             0.1, 0.2, 0.3, 0.4, 0.5,
                             0.6, 0.7, 0.8, 0.9, 1.0],

        "trading_window_days": "1D",
        "sliding_window_step": "1D",
        "lookback_window": 60,
        # Optional per-frequency lookback override.
        # Falls back to lookback_window when a frequency key is missing.
        "lookback_window_map": {
            "1m": 60,
            "5m": 30,
            "30m": 10,
        },

        # ----- Episode Configuration (NEW) -----
        # Fixed-length episodes ensure uniform data coverage and consistent gradient signal
        "episode_length": None,  # None = auto-calculate from frequency (1 week worth of bars)
        "episode_length_map": {   # Bars per episode by frequency (≈1 week each)
            "1m":  10080,         # 7 * 24 * 60
            "5m":  2016,          # 7 * 24 * 12
            "15m": 672,           # 7 * 24 * 4
            "30m": 336,           # 7 * 24 * 2
            "1h":  168,           # 7 * 24
            "1d":  7,
        },

        # ----- Transaction Costs -----
        "transaction_costs": {
            "maker_fee_structures": [0.0, 1.7, 5],
            "taker_bps": 0,
            "slippage_bps": 0,
            # Vol-dependent slippage parameters (NEW)
            "base_slippage_bps": 0.5,        # Base slippage in bps
            "vol_slippage_multiplier": 2.0,   # How much volatility amplifies slippage
        },

        # ----- Reward Configuration -----
        "reward": {
            "type": "sharpe",                 # "sharpe" (differential Sharpe) or "utility" (quadratic)
            "sharpe_eta": 0.01,               # Adaptation rate for differential Sharpe ratio
            "lambda_basic": 0.01,
            "lambda_utility": 6,
            "reward_clip": 3.0,
            "reward_scale": 10000,            # Scale factor for raw returns before reward computation
        },

        "seed": 42,
    },

    "SPLITS": {
        "data_start": "2024-05-01",
        "data_end": "2025-04-30",

        "train": ["2024-05-01 00:00:00", "2024-12-31 23:59:59"],
        "val":   ["2025-01-01 00:00:00", "2025-02-28 23:59:59"],
        "test":  ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],
    },

    "RL": {
        # ===== General =====
        "algorithm": "PPO",
        "timesteps": 2e5,
        "policy": "MlpPolicy",  # Will be overridden to use CNN extractor
        "gamma": 0.995,
        "learning_rate": 3e-4,
        "batch_size": 128,

        # ===== PPO =====
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": 0.2,        # NEW: clip value function too
        "n_steps": 2048,
        "n_epochs": 10,
        "ent_coef": 0.05,            # Start moderate, anneal via callback
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": True,
        "sde_sample_freq": -1,

        # ===== Off-Policy (SAC / DQN) =====
        "buffer_size": 200000,
        "learning_starts": 5000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef_SAC": "auto",
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1,

        # ===== Feature Extractor (NEW) =====
        "feature_extractor": "cnn",   # "cnn" or "mlp"
        "cnn_features_dim": 128,      # Output dim of CNN feature extractor
        "net_arch_pi": [64, 64],      # Policy network hidden layers
        "net_arch_vf": [64, 64],      # Value function hidden layers

        # ===== VecNormalize (NEW) =====
        "normalize_obs": True,
        "normalize_reward": True,
        "clip_obs": 10.0,
    },

    "EVAL": {
        "plots": True,
        "reports_dir": "./reports",
        "frequency": 14400,
        "n_eval_episodes": 3,
        "save_freq": 144000,
        "action_std_threshold": 0.01,
        "action_extreme_threshold": 1,
    },

    "IO": {
        "models_dir": "./models",
        "tb_logdir": "./tb_logs",
    },
}


# ============================================================================
# Annualization factors for different timeframes
# ============================================================================
ANNUALIZATION = {
    "1m":  365 * 24 * 60,
    "5m":  365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h":  365 * 24,
    "1d":  365,
}