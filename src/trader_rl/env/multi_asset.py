"""Multi-asset trading environment compatible with Gymnasium."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

from .config import EnvConfig
from trader_rl.features.indicators import ema, rsi, zscore


class MultiAssetTradingEnv(gym.Env):
    """A simplified yet feature-rich multi-asset trading environment."""

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, symbols: Optional[List[str]] = None, cfg: EnvConfig = EnvConfig()):
        super().__init__()
        required_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise AssertionError(
                "DataFrame must include timestamp,symbol,open,high,low,close,volume columns"
            )
        self.cfg = cfg

        self.symbols = symbols if symbols is not None else sorted(df["symbol"].unique().tolist())
        self.N = len(self.symbols)
        self._build_panel(df)

        self.T = self.cfg.window_size
        self.F = self.features.shape[-1]
        self.P = 5
        self.G = 3

        self.observation_space = spaces.Dict(
            {
                "time": spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.T, self.F), dtype=np.float32),
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.P), dtype=np.float32),
                "global": spaces.Box(low=-np.inf, high=np.inf, shape=(self.G,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Dict(
            {
                "disc": spaces.MultiDiscrete(np.array([3] * self.N, dtype=np.int64)),
                "target": spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32),
            }
        )

        self._rng = np.random.RandomState(42)
        self._reset_state()

    # ---------- подготовка панели ----------
    def _build_panel(self, df: pd.DataFrame) -> None:
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise AssertionError("Missing required columns for panel construction")

        df = df.copy()

        if not is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, infer_datetime_format=True)
        else:
            if is_datetime64tz_dtype(df["timestamp"]):
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        if "spread_bp" in df.columns:
            df["spread_bp"] = pd.to_numeric(df["spread_bp"], errors="coerce")
            agg["spread_bp"] = "mean"
        if "borrow_fee_bp_day" in df.columns:
            df["borrow_fee_bp_day"] = pd.to_numeric(df["borrow_fee_bp_day"], errors="coerce")
            agg["borrow_fee_bp_day"] = "mean"

        self.symbols = self.symbols if hasattr(self, "symbols") and self.symbols else sorted(df["symbol"].unique().tolist())
        self.N = len(self.symbols)

        sym_groups: Dict[str, pd.DataFrame] = {}
        idx_map: Dict[str, pd.DatetimeIndex] = {}
        for s in self.symbols:
            g = df[df["symbol"] == s].copy().sort_values("timestamp")
            g = (
                g.groupby("timestamp", as_index=False)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                        **({"spread_bp": "mean"} if "spread_bp" in g.columns else {}),
                        **({"borrow_fee_bp_day": "mean"} if "borrow_fee_bp_day" in g.columns else {}),
                    }
                )
                .sort_values("timestamp")
            )

            idx = pd.DatetimeIndex(g["timestamp"])
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            g["timestamp"] = idx

            g = g.dropna(subset=["open", "high", "low", "close", "volume"])
            if g.empty:
                raise ValueError(f"No valid OHLCV bars remain after aggregation for symbol {s}")

            sym_groups[s] = g
            idx_map[s] = pd.DatetimeIndex(g["timestamp"])

        common_idx = None
        for s in self.symbols:
            common_idx = idx_map[s] if common_idx is None else common_idx.intersection(idx_map[s])

        if common_idx is None or len(common_idx) < self.cfg.window_size + 2:
            lens = {s: len(idx_map[s]) for s in self.symbols}
            pairs = []
            for i in range(len(self.symbols)):
                for j in range(i + 1, len(self.symbols)):
                    si, sj = self.symbols[i], self.symbols[j]
                    pairs.append((si, sj, len(idx_map[si].intersection(idx_map[sj]))))
            raise ValueError(
                "Insufficient overlap between instruments: "
                f"|common|={0 if common_idx is None else len(common_idx)}, window={self.cfg.window_size}. "
                f"Lengths: {lens}. Pairwise intersections: {pairs[:6]}"
            )

        self.timestamps = common_idx.sort_values()
        L = len(self.timestamps)

        O = np.zeros((self.N, L), dtype=np.float64)
        H = np.zeros((self.N, L), dtype=np.float64)
        Lo = np.zeros((self.N, L), dtype=np.float64)
        C = np.zeros((self.N, L), dtype=np.float64)
        V = np.zeros((self.N, L), dtype=np.float64)
        maybe_spread = np.full((self.N, L), np.nan, dtype=np.float64)
        maybe_borrow = np.full((self.N, L), np.nan, dtype=np.float64)

        for i, s in enumerate(self.symbols):
            g = sym_groups[s].set_index("timestamp").reindex(self.timestamps)
            mask_nan = g[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
            if mask_nan.any():
                bad = g[mask_nan].index[:5]
                raise ValueError(
                    f"Unexpected NaNs for symbol {s} after aligning to common grid. Sample timestamps: {list(bad)}"
                )
            O[i] = g["open"].values
            H[i] = g["high"].values
            Lo[i] = g["low"].values
            C[i] = g["close"].values
            V[i] = g["volume"].values
            if "spread_bp" in g.columns:
                maybe_spread[i] = g["spread_bp"].astype(float).values
            if "borrow_fee_bp_day" in g.columns:
                maybe_borrow[i] = g["borrow_fee_bp_day"].astype(float).values

        feats = []
        for i in range(self.N):
            close = C[i]
            high = H[i]
            low = Lo[i]
            vol = V[i]

            ret = np.log(close / np.clip(np.roll(close, 1), 1e-12, None))
            ret[0] = 0.0
            hlrng = (high - low) / np.clip(np.roll(close, 1), 1e-12, None)
            hlrng[0] = 0.0

            volz = zscore(vol, window=64)

            ema12 = ema(close, span=12)
            ema26 = ema(close, span=26)
            macd = ema12 - ema26

            rsi14 = rsi(close, period=14)

            adv = pd.Series(vol * close).rolling(20).mean().fillna(method="bfill").values

            if self.cfg.feature_set == "minimal":
                f = np.stack([ret, hlrng, volz], axis=-1)
            else:
                f = np.stack(
                    [
                        ret,
                        hlrng,
                        volz,
                        macd,
                        rsi14,
                        ema12 / np.clip(close, 1e-12, None),
                        ema26 / np.clip(close, 1e-12, None),
                    ],
                    axis=-1,
                )

            f = np.concatenate([f, adv[:, None]], axis=-1)
            feats.append(f)

        self.opens = O
        self.highs = H
        self.lows = Lo
        self.closes = C
        self.volumes = V
        self.adv = np.stack([x[:, -1] for x in feats], axis=0)
        self.features = np.stack(feats, axis=0)
        self.spread_bp_ts = maybe_spread
        self.borrow_fee_bp_ts = maybe_borrow

        ts = self.timestamps
        ns = ts.asi8.astype(np.int64)
        dt_sec = (ns[1:] - ns[:-1]) / 1e9
        self.dt_days = np.concatenate([[dt_sec[0] / 86400.0], dt_sec / 86400.0])

    def _reset_state(self) -> None:
        self.t = self.cfg.window_size
        self.qty = np.zeros(self.N, dtype=np.float64)
        self.entry_px = np.zeros(self.N, dtype=np.float64)
        self.entry_t = np.zeros(self.N, dtype=np.int64)
        self.cash = float(self.cfg.initial_equity)
        self.equity = float(self.cfg.initial_equity)
        self.done_flag = False
        self.info_last: Dict[str, float] = {}

    def _prices(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.opens[:, t], self.highs[:, t], self.closes[:, t]

    def _midprice(self, t: int) -> np.ndarray:
        return self.closes[:, t].astype(np.float64)

    def _obs(self) -> Dict[str, np.ndarray]:
        t = self.t
        idx0 = t - self.T
        x_time = self.features[:, idx0:t, :].astype(np.float32)
        prices = self._midprice(t)
        pos_value = self.qty * prices
        pos_fraction = pos_value / max(self.equity, 1e-12)
        is_open = (self.qty != 0).astype(np.float32)
        age = np.where(self.qty != 0, (t - self.entry_t), 0).astype(np.float32)
        unreal_pnl_bp = np.zeros(self.N, dtype=np.float32)
        mask_open = self.qty != 0
        unreal_pnl_bp[mask_open] = (
            (
                prices[mask_open]
                / np.clip(self.entry_px[mask_open], 1e-12, None)
                - 1.0
            )
            * 10_000.0
            * np.sign(self.qty[mask_open])
        )

        x_pos = np.stack(
            [
                is_open.astype(np.float32),
                pos_fraction.astype(np.float32),
                (np.divide(self.entry_px, prices, out=np.zeros_like(self.entry_px), where=prices > 0)).astype(
                    np.float32
                ),
                age.astype(np.float32),
                unreal_pnl_bp.astype(np.float32),
            ],
            axis=-1,
        )

        gross = float(np.sum(np.abs(pos_value)))
        net = float(np.sum(pos_value))
        g = np.array(
            [
                float(self.cash / max(self.equity, 1e-12)),
                float(gross / max(self.equity, 1e-12)),
                float(net / max(self.equity, 1e-12)),
            ],
            dtype=np.float32,
        )

        return {"time": x_time, "pos": x_pos.astype(np.float32), "global": g}

    def _execute(self, disc: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        t = self.t
        price = self._midprice(t)
        equity_before = self._mark_to_market(price)
        equity = equity_before

        desired_frac = target.copy()
        target_notional = desired_frac * equity
        target_qty = np.divide(target_notional, price, out=np.zeros_like(price), where=price > 0)

        order_qty = target_qty - self.qty

        new_qty = self.qty + order_qty
        gross_after = float(np.sum(np.abs(new_qty * price)))
        max_gross = self.cfg.max_leverage * equity

        scale = 1.0
        if gross_after > max_gross and gross_after > 0:
            excess_ratio = max_gross / gross_after
            scale = excess_ratio
            order_qty = order_qty * scale
            new_qty = self.qty + order_qty

        spread_bp = np.where(np.isnan(self.spread_bp_ts[:, t]), self.cfg.spread_bp, self.spread_bp_ts[:, t])
        half_spread_bp = spread_bp / 2.0
        notional_trade = np.abs(order_qty) * price
        adv = np.maximum(self.adv[:, t], 1.0)
        slippage_bp = np.clip(self.cfg.impact_coef * (notional_trade / adv) * 10_000.0, 0.0, 200.0)
        side = np.sign(order_qty)
        fill_mult = 1.0 + side * ((half_spread_bp + slippage_bp) / 10_000.0)
        fill_price = price * fill_mult

        trade_cash = np.sum(order_qty * fill_price)
        commissions = np.sum((np.abs(order_qty * fill_price)) * (self.cfg.commission_bp / 10_000.0))
        self.cash -= trade_cash + commissions
        self.qty = new_qty

        dt_day = float(self.dt_days[t])
        short_notional = np.sum(np.abs(np.minimum(self.qty, 0.0) * price))
        borrow_cost = short_notional * (self.cfg.borrow_fee_bp_day / 10_000.0) * dt_day
        self.cash -= borrow_cost

        turnover = float(np.sum(np.abs(order_qty * fill_price)) / max(equity, 1e-12))

        opened_mask = np.isclose(self.qty - order_qty, 0.0) & (~np.isclose(self.qty, 0.0))
        self.entry_px[opened_mask] = fill_price[opened_mask]
        self.entry_t[opened_mask] = t

        return float(commissions), float(borrow_cost), turnover

    def _mark_to_market(self, price_now: np.ndarray) -> float:
        return float(self.cash + np.sum(self.qty * price_now))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._reset_state()
        obs = self._obs()
        return obs, {}

    def step(self, action: Dict[str, np.ndarray]):
        if self.done_flag:
            raise AssertionError("Episode already done. Call reset().")
        disc = np.asarray(action["disc"], dtype=np.int64)
        target = np.asarray(action["target"], dtype=np.float32)
        target = np.clip(target, -1.0, 1.0)

        price_t = self._midprice(self.t)
        equity_t = self._mark_to_market(price_t)
        current_frac = np.divide(self.qty * price_t, max(equity_t, 1e-12))
        hold_mask = disc == 0
        target[hold_mask] = current_frac[hold_mask]

        commissions, borrow_cost, turnover = self._execute(disc, target)

        t_next = self.t + 1
        price_next = self._midprice(t_next)
        equity_next = self._mark_to_market(price_next)

        reward = equity_next - equity_t
        self.equity = equity_next
        self.t = t_next

        terminated = self.t >= len(self.timestamps) - 1
        truncated = False
        self.done_flag = terminated or truncated

        self.info_last = {
            "equity": self.equity,
            "cash": self.cash,
            "commissions": commissions,
            "borrow_cost": borrow_cost,
            "turnover": turnover,
        }
        obs = self._obs() if not self.done_flag else self._obs()
        return obs, float(reward), terminated, truncated, self.info_last

    def render(self):
        pass
