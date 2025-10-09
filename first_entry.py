# rl_trader_env_ppo.py
import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


# =============== УТИЛИТЫ ФИЧЕЙ ===============

def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    # series: 1D close
    diff = np.diff(series, prepend=series[0])
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    roll_up = pd.Series(up).rolling(period).mean().values
    roll_dn = pd.Series(dn).rolling(period).mean().values
    rs = np.divide(roll_up, roll_dn + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[np.isnan(out)] = 50.0
    return out

def ema(series: np.ndarray, span: int = 12) -> np.ndarray:
    return pd.Series(series).ewm(span=span, adjust=False).mean().values

def zscore(x: np.ndarray, window: int = 64) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(window).mean().values
    v = s.rolling(window).std().values
    z = (x - m) / (v + 1e-8)
    z[np.isnan(z)] = 0.0
    return z

# =============== СРЕДА ===============

@dataclass
class EnvConfig:
    window_size: int = 64
    initial_equity: float = 1_000_000.0
    commission_bp: float = 2.0          # комиссия за сделку (bps)
    borrow_fee_bp_day: float = 20.0     # шорт-фии bps/день
    spread_bp: float = 10.0             # средний спред bps (если нет bid/ask)
    impact_coef: float = 2e-4           # коэффициент ударного слиппеджа ~ size/ADV
    max_leverage: float = 2.0
    gamma: float = 0.999
    feature_set: str = "minimal"        # "minimal" или "tech"

class MultiAssetTradingEnv(gym.Env):
    """
    Наблюдение: Dict{
        'time':   [N, T, F] float32
        'pos':    [N, P]    float32
        'global': [G]       float32
    }
    Действие: Dict{
        'disc':   MultiDiscrete([3]*N)  # 0 Hold, 1 Buy, 2 Sell
        'target': Box([-1, 1], (N,))
    }
    Вознаграждение: dEquity (в валюте счёта).
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, symbols: Optional[List[str]] = None, cfg: EnvConfig = EnvConfig()):
        super().__init__()
        assert {'timestamp','symbol','open','high','low','close','volume'}.issubset(df.columns), \
            "В DataFrame должны быть колонки: timestamp,symbol,open,high,low,close,volume"
        self.cfg = cfg

        # Подготовим панель (общая временная сетка для всех тикеров)
        self.symbols = symbols if symbols is not None else sorted(df['symbol'].unique().tolist())
        self.N = len(self.symbols)
        self._build_panel(df)

        self.T = self.cfg.window_size
        self.F = self.features.shape[-1]
        self.P = 5  # pos features: is_open, pos_fraction, entry_price_rel, time_in_pos, unreal_pnl_bp
        self.G = 3  # global: cash_frac, gross_exp, net_exp

        # spaces
        self.observation_space = spaces.Dict({
            "time":   spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.T, self.F), dtype=np.float32),
            "pos":    spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.P), dtype=np.float32),
            "global": spaces.Box(low=-np.inf, high=np.inf, shape=(self.G,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            "disc":   spaces.MultiDiscrete(np.array([3]*self.N, dtype=np.int64)),
            "target": spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32),
        })

        self._rng = np.random.RandomState(42)
        self._reset_state()

    # ---------- подготовка панели ----------
    def _build_panel(self, df: pd.DataFrame):
        required = {'timestamp','symbol','open','high','low','close','volume'}
        assert required.issubset(df.columns), "Нужны колонки: timestamp,symbol,open,high,low,close,volume"

        df = df.copy()

        # 1) Приводим timestamp к UTC, устойчиво к tz-aware/naive
        if not is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, infer_datetime_format=True)
        else:
            if is_datetime64tz_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

        # 2) Явно к float для цен/объёмов
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3) Агрегация по timestamp для устранения дубликатов
        #    OHLCV: open=first, high=max, low=min, close=last, volume=sum
        agg = {
            'open': 'first',
            'high': 'max',
            'low':  'min',
            'close':'last',
            'volume':'sum'
        }
        if 'spread_bp' in df.columns:
            df['spread_bp'] = pd.to_numeric(df['spread_bp'], errors='coerce')
            agg['spread_bp'] = 'mean'
        if 'borrow_fee_bp_day' in df.columns:
            df['borrow_fee_bp_day'] = pd.to_numeric(df['borrow_fee_bp_day'], errors='coerce')
            # можно 'last' или 'mean' — берём усреднение
            agg['borrow_fee_bp_day'] = 'mean'

        # 4) По каждому символу — агрегируем и фильтруем только «полные бары»
        self.symbols = self.symbols if hasattr(self, 'symbols') and self.symbols else sorted(df['symbol'].unique().tolist())
        self.N = len(self.symbols)

        sym_groups = {}
        idx_map = {}
        for s in self.symbols:
            g = df[df['symbol'] == s].copy().sort_values('timestamp')

            # агрегирование дубликатов по timestamp, как у вас выше
            # (если уже есть — оставьте; если нет, добавьте)
            g = (g.groupby('timestamp', as_index=False)
                .agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum',
                        **({'spread_bp':'mean'} if 'spread_bp' in g.columns else {}),
                        **({'borrow_fee_bp_day':'mean'} if 'borrow_fee_bp_day' in g.columns else {})})
                .sort_values('timestamp'))

            # приводим индекс к tz-aware UTC (на всякий случай)
            idx = pd.DatetimeIndex(g['timestamp'])
            if idx.tz is None:
                idx = idx.tz_localize('UTC')
            else:
                idx = idx.tz_convert('UTC')
            g['timestamp'] = idx

            # убираем неполные бары
            g = g.dropna(subset=['open','high','low','close','volume'])
            if g.empty:
                raise ValueError(f"После агрегации у символа {s} нет полных баров OHLCV")

            sym_groups[s] = g
            idx_map[s] = pd.DatetimeIndex(g['timestamp'])  # tz-aware, UTC

        # Пересечение только через DatetimeIndex.intersection (без .values и set)
        common_idx = None
        for s in self.symbols:
            common_idx = idx_map[s] if common_idx is None else common_idx.intersection(idx_map[s])

        # Проверка объёма пересечения
        if common_idx is None or len(common_idx) < self.cfg.window_size + 2:
            lens = {s: len(idx_map[s]) for s in self.symbols}
            pairs = []
            for i in range(len(self.symbols)):
                for j in range(i+1, len(self.symbols)):
                    si, sj = self.symbols[i], self.symbols[j]
                    pairs.append((si, sj, len(idx_map[si].intersection(idx_map[sj]))))
            raise ValueError(
                f"Слишком мало общих баров: |common|={0 if common_idx is None else len(common_idx)}, "
                f"окно={self.cfg.window_size}. Размеры: {lens}. Парные пересечения (часть): {pairs[:6]}"
            )

        # Храним timestamps как tz-aware DatetimeIndex (UTC)
        self.timestamps = common_idx.sort_values()  # DatetimeIndex(UTC)
        L = len(self.timestamps)

        # 6) Формируем панельные массивы без NaN
        O = np.zeros((self.N, L), dtype=np.float64)
        H = np.zeros((self.N, L), dtype=np.float64)
        Lo = np.zeros((self.N, L), dtype=np.float64)
        C = np.zeros((self.N, L), dtype=np.float64)
        V = np.zeros((self.N, L), dtype=np.float64)
        maybe_spread = np.full((self.N, L), np.nan, dtype=np.float64)
        maybe_borrow = np.full((self.N, L), np.nan, dtype=np.float64)

        for i, s in enumerate(self.symbols):
            g = sym_groups[s].set_index('timestamp').reindex(self.timestamps)
            # после «common» пропусков быть не должно — проверим и дадим полезную диагностику
            mask_nan = g[['open','high','low','close','volume']].isna().any(axis=1)
            if mask_nan.any():
                bad = g[mask_nan].index[:5]
                raise ValueError(
                    f"Неожиданные NaN у {s} после reindex на общий грид. Примеры ts: {list(bad)}. "
                    f"Проверьте наличие нескольких таймфреймов/смешанных книжек."
                )
            O[i]  = g['open'].values
            H[i]  = g['high'].values
            Lo[i] = g['low'].values
            C[i]  = g['close'].values
            V[i]  = g['volume'].values
            if 'spread_bp' in g.columns:
                maybe_spread[i] = g['spread_bp'].astype(float).values
            if 'borrow_fee_bp_day' in g.columns:
                maybe_borrow[i] = g['borrow_fee_bp_day'].astype(float).values

        # 7) Фичи по каждому символу
        feats = []
        for i in range(self.N):
            close = C[i]
            high  = H[i]
            low   = Lo[i]
            vol   = V[i]

            ret   = np.log(close / np.clip(np.roll(close, 1), 1e-12, None)); ret[0] = 0.0
            hlrng = (high - low) / np.clip(np.roll(close,1), 1e-12, None); hlrng[0] = 0.0

            # простая z-нормализация объёма (скользящее окно)
            s = pd.Series(vol)
            m = s.rolling(64).mean().values
            v = s.rolling(64).std().values
            volz = (vol - m) / (v + 1e-8)
            volz[np.isnan(volz)] = 0.0

            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
            macd  = ema12 - ema26

            # RSI(14)
            diff = np.diff(close, prepend=close[0])
            up = np.clip(diff, 0, None)
            dn = np.clip(-diff, 0, None)
            roll_up = pd.Series(up).rolling(14).mean().values
            roll_dn = pd.Series(dn).rolling(14).mean().values
            rs = np.divide(roll_up, roll_dn + 1e-12)
            rsi14 = 100.0 - (100.0 / (1.0 + rs))
            rsi14[np.isnan(rsi14)] = 50.0

            adv   = pd.Series(vol*close).rolling(20).mean().fillna(method='bfill').values

            if self.cfg.feature_set == "minimal":
                f = np.stack([ret, hlrng, volz], axis=-1)
            else:
                f = np.stack([
                    ret, hlrng, volz, macd, rsi14,
                    ema12/np.clip(close,1e-12,None),
                    ema26/np.clip(close,1e-12,None)
                ], axis=-1)

            f = np.concatenate([f, adv[:, None]], axis=-1)  # добавим ADV
            feats.append(f)

        self.opens  = O
        self.highs  = H
        self.lows   = Lo
        self.closes = C
        self.volumes= V
        self.adv    = np.stack([x[:, -1] for x in feats], axis=0)
        self.features = np.stack(feats, axis=0)
        self.spread_bp_ts = maybe_spread
        self.borrow_fee_bp_ts = maybe_borrow  # если NaN — используем cfg.borrow_fee_bp_day

        # 8) Δt в днях (ASI8 — надёжно и для tz-aware)
        ts = self.timestamps                       # уже DatetimeIndex(UTC)
        ns = ts.asi8.astype(np.int64)              # наносекунды от эпохи
        dt_sec = (ns[1:] - ns[:-1]) / 1e9
        self.dt_days = np.concatenate([[dt_sec[0]/86400.0], dt_sec/86400.0])



    # ---------- служебное состояние/сброс ----------
    def _reset_state(self):
        self.t = self.cfg.window_size  # первый индекс, где доступно окно
        self.qty = np.zeros(self.N, dtype=np.float64)  # количества акций (может быть <0 для шорта)
        self.entry_px = np.zeros(self.N, dtype=np.float64)
        self.entry_t  = np.zeros(self.N, dtype=np.int64)
        self.cash = float(self.cfg.initial_equity)
        self.equity = float(self.cfg.initial_equity)
        self.done_flag = False
        self.info_last = {}

    def _prices(self, t: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        return self.opens[:, t], self.highs[:, t], self.closes[:, t]

    def _midprice(self, t: int) -> np.ndarray:
        # без стакана — используем close как mid
        return self.closes[:, t].astype(np.float64)

    # ---------- построение наблюдения ----------
    def _obs(self):
        t = self.t
        idx0 = t - self.T
        x_time = self.features[:, idx0:t, :].astype(np.float32)  # [N,T,F]
        prices = self._midprice(t)
        pos_value = self.qty * prices
        pos_fraction = pos_value / max(self.equity, 1e-12)
        is_open = (self.qty != 0).astype(np.float32)
        age = np.where(self.qty != 0, (t - self.entry_t), 0).astype(np.float32)
        unreal_pnl_bp = np.zeros(self.N, dtype=np.float32)
        mask_open = self.qty != 0
        unreal_pnl_bp[mask_open] = ((prices[mask_open] / np.clip(self.entry_px[mask_open], 1e-12, None)) - 1.0) * 10_000.0 * np.sign(self.qty[mask_open])

        x_pos = np.stack([
            is_open.astype(np.float32),
            pos_fraction.astype(np.float32),
            (np.divide(self.entry_px, prices, out=np.zeros_like(self.entry_px), where=prices>0)).astype(np.float32),
            age.astype(np.float32),
            unreal_pnl_bp.astype(np.float32),
        ], axis=-1)  # [N,P]

        gross = float(np.sum(np.abs(pos_value)))
        net   = float(np.sum(pos_value))
        g = np.array([
            float(self.cash / max(self.equity, 1e-12)),
            float(gross / max(self.equity, 1e-12)),
            float(net   / max(self.equity, 1e-12)),
        ], dtype=np.float32)

        return {"time": x_time, "pos": x_pos.astype(np.float32), "global": g}

    # ---------- исполнение приказов ----------
    def _execute(self, disc: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        """
        Возвращает: commissions, borrow_cost, turnover
        """
        t = self.t
        price = self._midprice(t)  # [N]
        equity_before = self._mark_to_market(price)  # equity до сделок
        equity = equity_before

        # Маска Buy/Sell/Hold -> желаемый target
        desired_frac = target.copy()
        # Для Hold оставим текущую долю (через qty); для Buy/Sell не принуждаем знак (контроль оставим на target)
        # Превратим долю в целевой notional и quantity:
        target_notional = desired_frac * equity  # [N]
        target_qty = np.divide(target_notional, price, out=np.zeros_like(price), where=price>0)

        order_qty = target_qty - self.qty  # [N]

        # ограничение по левереджу: смоделируем исполнение, оценим выставку, при необходимости масштабируем
        # Оценка notional после исполнения:
        new_qty = self.qty + order_qty
        gross_after = float(np.sum(np.abs(new_qty * price)))
        max_gross = self.cfg.max_leverage * equity

        scale = 1.0
        if gross_after > max_gross and gross_after > 0:
            # масштабируем весь приращенный заказ
            excess_ratio = max_gross / gross_after
            scale = excess_ratio
            order_qty = order_qty * scale
            new_qty = self.qty + order_qty

        # прайс исполнения с половиной спрэда и ударным слиппеджем
        spread_bp = np.where(np.isnan(self.spread_bp_ts[:, t]), self.cfg.spread_bp, self.spread_bp_ts[:, t])
        half_spread_bp = spread_bp / 2.0
        notional_trade = np.abs(order_qty) * price
        adv = np.maximum(self.adv[:, t], 1.0)
        slippage_bp = np.clip(self.cfg.impact_coef * (notional_trade / adv) * 10_000.0, 0.0, 200.0)  # cap 200 bps
        side = np.sign(order_qty)
        fill_mult = 1.0 + side * ( (half_spread_bp + slippage_bp)/10_000.0 )
        fill_price = price * fill_mult

        trade_cash = np.sum(order_qty * fill_price)  # >0 если покупаем больше чем продаём (расход кэша)
        commissions = np.sum((np.abs(order_qty * fill_price)) * (self.cfg.commission_bp/10_000.0))
        self.cash -= trade_cash + commissions
        self.qty = new_qty

        # шорт-фии до следующего бара
        dt_day = float(self.dt_days[t])
        short_notional = np.sum(np.abs(np.minimum(self.qty, 0.0) * price))
        borrow_cost = short_notional * (self.cfg.borrow_fee_bp_day/10_000.0) * dt_day
        self.cash -= borrow_cost

        # оборот (turnover) в долях портфеля
        turnover = float(np.sum(np.abs(order_qty * fill_price)) / max(equity, 1e-12))

        # обновим entry_px/entry_t там, где позиция была открыта из нуля
        opened_mask = (np.isclose(self.qty - order_qty, 0.0) & (~np.isclose(self.qty, 0.0)))
        self.entry_px[opened_mask] = fill_price[opened_mask]
        self.entry_t[opened_mask]  = t

        return float(commissions), float(borrow_cost), turnover

    def _mark_to_market(self, price_now: np.ndarray) -> float:
        return float(self.cash + np.sum(self.qty * price_now))

    # ---------- API Gym ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._reset_state()
        obs = self._obs()
        return obs, {}

    def step(self, action: Dict[str, np.ndarray]):
        assert not self.done_flag, "Episode already done. Call reset()."
        disc = np.asarray(action["disc"], dtype=np.int64)
        target = np.asarray(action["target"], dtype=np.float32)
        target = np.clip(target, -1.0, 1.0)

        # применяем Buy/Sell/Hold: мягкая логика — Hold: тянем target к текущей доле
        price_t = self._midprice(self.t)
        equity_t = self._mark_to_market(price_t)
        current_frac = np.divide(self.qty * price_t, max(equity_t,1e-12))
        # если Hold -> целевая доля = текущая
        hold_mask = (disc == 0)
        target[hold_mask] = current_frac[hold_mask]

        # Исполнение + издержки до t+1
        commissions, borrow_cost, turnover = self._execute(disc, target)

        # Переоценка на следующем баре
        t_next = self.t + 1
        price_next = self._midprice(t_next)
        equity_next = self._mark_to_market(price_next)

        reward = equity_next - equity_t  # dEquity
        self.equity = equity_next
        self.t = t_next

        terminated = (self.t >= len(self.timestamps)-1)
        truncated = False
        self.done_flag = terminated or truncated

        self.info_last = {
            "equity": self.equity,
            "cash": self.cash,
            "commissions": commissions,
            "borrow_cost": borrow_cost,
            "turnover": turnover,
        }
        obs = self._obs() if not self.done_flag else self._obs()  # финальное наблюдение допустимо вернуть
        return obs, float(reward), terminated, truncated, self.info_last

    def render(self):
        pass

# =============== МОДЕЛЬ (минимальная, per-instrument, без кросс-внимания для простоты) ===============

class TimeEncoder(nn.Module):
    def __init__(self, f_in: int, d_out: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(f_in, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.proj  = nn.Linear(64, d_out)
        self.norm  = nn.LayerNorm(d_out)

    def forward(self, x):  # x: [B,N,T,F]
        B,N,T,Ft = x.shape
        x = x.reshape(B*N, T, Ft).transpose(1,2)    # [B*N, F, T]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.mean(dim=-1)                          # [B*N, 64]
        x = self.proj(x)
        x = self.norm(x)
        return x.reshape(B, N, -1)                  # [B,N,d_out]

class PolicyNet(nn.Module):
    def __init__(self, f_time: int, p_in: int, g_in: int, d_model: int = 128):
        super().__init__()
        self.time_enc = TimeEncoder(f_time, d_out=64)
        self.pos_mlp  = nn.Sequential(nn.Linear(p_in, 64), nn.GELU(), nn.Linear(64, 32))
        self.g_proj   = nn.Linear(g_in, 32)

        self.backbone = nn.Sequential(
            nn.Linear(64+32+32, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.head_disc = nn.Linear(d_model, 3)   # logits per instrument
        self.head_mean = nn.Linear(d_model, 1)   # mean for tanh-Normal
        self.log_std   = nn.Parameter(torch.zeros(1))  # shared log_std for continuous
        # Критик — агрегируем средним по бумагам
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_time, x_pos, x_global):
        B,N,T,Ft = x_time.shape
        h_time = self.time_enc(x_time)           # [B,N,64]
        h_pos  = self.pos_mlp(x_pos)             # [B,N,32]
        g = F.gelu(self.g_proj(x_global))        # [B,32]
        g = g.unsqueeze(1).repeat(1,N,1)         # [B,N,32]
        z = torch.cat([h_time, h_pos, g], dim=-1) # [B,N,128]
        z = self.backbone(z)                     # [B,N,d_model]

        logits = self.head_disc(z)               # [B,N,3]
        mean   = torch.tanh(self.head_mean(z)).squeeze(-1)  # [-1,1], [B,N]
        value  = self.value_head(z.mean(dim=1)).squeeze(-1) # [B]
        std    = torch.exp(self.log_std) + 1e-6
        return logits, mean, std, value

    def act(self, obs):
        x_time, x_pos, x_global = obs["time"], obs["pos"], obs["global"]
        logits, mean, std, value = self.forward(x_time, x_pos, x_global)
        B,N,_ = logits.shape
        # дискретная часть
        cat = Categorical(logits=logits)     # факторизованно по N, вернёт [B,N]
        disc = cat.sample()
        logp_disc = cat.log_prob(disc).sum(dim=-1)  # суммируем по инструментам

        # непрерывная часть (tanh-Normal)
        base = Normal(mean, std)
        y = base.rsample()
        cont = torch.tanh(y)
        # логправ с якобианом tanh
        logp_cont = base.log_prob(y) - torch.log(1.0 - cont.pow(2) + 1e-8)
        logp_cont = logp_cont.sum(dim=-1)  # суммируем по инструментам

        return disc, cont, logp_disc + logp_cont, value

    def evaluate_actions(self, obs, disc_actions, cont_actions):
        x_time, x_pos, x_global = obs["time"], obs["pos"], obs["global"]
        logits, mean, std, value = self.forward(x_time, x_pos, x_global)
        # disc
        cat = Categorical(logits=logits)
        logp_disc = cat.log_prob(disc_actions).sum(dim=-1)
        entropy_disc = cat.entropy().sum(dim=-1)

        # cont: обратим tanh -> atanh
        atanh = 0.5 * (torch.log1p(cont_actions + 1e-8) - torch.log1p(-cont_actions + 1e-8))
        base = Normal(mean, std)
        logp_cont = base.log_prob(atanh) - torch.log(1.0 - cont_actions.pow(2) + 1e-8)
        logp_cont = logp_cont.sum(dim=-1)
        entropy_cont = (0.5 * (1.0 + math.log(2*math.pi)) + torch.log(std)).sum(dim=-1)  # энтропия Normal без учёта tanh

        logp = logp_disc + logp_cont
        entropy = entropy_disc + entropy_cont
        return logp, entropy, value

# =============== PPO АГЕНТ ===============

@dataclass
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 6
    minibatch_size: int = 64
    rollout_steps: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"

class PPOAgent:
    def __init__(self, env: MultiAssetTradingEnv, cfg: PPOConfig):
        self.env = env
        obs_space = env.observation_space
        act_space = env.action_space
        # извлечём размеры из пробного obs
        obs,_ = env.reset()
        f_time = obs["time"].shape[-1]
        p_in   = obs["pos"].shape[-1]
        g_in   = obs["global"].shape[-1]
        self.N = obs["time"].shape[0]
        self.model = PolicyNet(f_time=f_time, p_in=p_in, g_in=g_in).to(cfg.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.device = cfg.device

    def _to_torch_obs(self, obs):
        return {
            "time":   torch.tensor(obs["time"], dtype=torch.float32, device=self.device).unsqueeze(0),  # [1,N,T,F]
            "pos":    torch.tensor(obs["pos"],  dtype=torch.float32, device=self.device).unsqueeze(0),
            "global": torch.tensor(obs["global"],dtype=torch.float32, device=self.device).unsqueeze(0),
        }

    def _stack_obs(self, obs_batch: Dict[str, List[np.ndarray]]):
        # превращаем списки obs в батч тензоров [B, ...]
        x_time = torch.tensor(np.stack(obs_batch["time"], axis=0), dtype=torch.float32, device=self.device)   # [B,N,T,F]
        x_pos  = torch.tensor(np.stack(obs_batch["pos"], axis=0),  dtype=torch.float32, device=self.device)   # [B,N,P]
        x_glob = torch.tensor(np.stack(obs_batch["global"], axis=0),dtype=torch.float32, device=self.device)  # [B,G]
        return {"time": x_time, "pos": x_pos, "global": x_glob}

    def collect_rollout(self):
        env = self.env
        obs, _ = env.reset()
        batch = {
            "obs_time": [], "obs_pos": [], "obs_global": [],
            "disc": [], "cont": [],
            "logp": [], "value": [],
            "reward": [], "done": [],
        }
        steps = 0
        while steps < self.cfg.rollout_steps:
            tobs = self._to_torch_obs(obs)
            with torch.no_grad():
                disc, cont, logp, value = self.model.act(tobs)
            # приведём к numpy
            disc_np = disc.squeeze(0).cpu().numpy().astype(np.int64)   # [N]
            cont_np = cont.squeeze(0).cpu().numpy().astype(np.float32)  # [N]
            action = {"disc": disc_np, "target": cont_np}
            next_obs, reward, term, trunc, info = env.step(action)

            batch["obs_time"].append(obs["time"])
            batch["obs_pos"].append(obs["pos"])
            batch["obs_global"].append(obs["global"])
            batch["disc"].append(disc_np)
            batch["cont"].append(cont_np)
            batch["logp"].append(logp.item())
            batch["value"].append(value.item())
            batch["reward"].append(reward)
            batch["done"].append(float(term or trunc))

            obs = next_obs
            steps += 1
            if term or trunc:
                obs, _ = env.reset()

        # векторизуем
        B = len(batch["reward"])
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        dones   = torch.tensor(batch["done"],   dtype=torch.float32, device=self.device)
        values  = torch.tensor(batch["value"],  dtype=torch.float32, device=self.device)

        # GAE
        with torch.no_grad():
            adv = torch.zeros(B, dtype=torch.float32, device=self.device)
            lastgaelam = 0.0
            next_value = 0.0
            for t in reversed(range(B)):
                nextnonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.cfg.gamma * next_value * nextnonterminal - values[t]
                lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                adv[t] = lastgaelam
                next_value = values[t]
            returns = adv + values

        # нормализуем advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # пакетируем
        obs_batch = {"time": batch["obs_time"], "pos": batch["obs_pos"], "global": batch["obs_global"]}
        obs_t = self._stack_obs(obs_batch)
        disc_t = torch.tensor(np.stack(batch["disc"], axis=0), dtype=torch.int64, device=self.device)  # [B,N]
        cont_t = torch.tensor(np.stack(batch["cont"], axis=0), dtype=torch.float32, device=self.device) # [B,N]
        logp_t = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        return obs_t, disc_t, cont_t, logp_t, values.detach(), returns.detach(), adv.detach()

    def update(self, obs_t, disc_t, cont_t, logp_old, values_old, returns, advantages):
        B = returns.shape[0]
        idx = np.arange(B)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, B, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = idx[start:end]
                mb_obs = {"time": obs_t["time"][mb_idx],
                          "pos": obs_t["pos"][mb_idx],
                          "global": obs_t["global"][mb_idx]}
                mb_disc = disc_t[mb_idx]
                mb_cont = cont_t[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_returns  = returns[mb_idx]
                mb_adv      = advantages[mb_idx]
                mb_values_old = values_old[mb_idx]

                logp, entropy, value = self.model.evaluate_actions(mb_obs, mb_disc, mb_cont)
                ratio = torch.exp(logp - mb_logp_old)

                pg1 = ratio * mb_adv
                pg2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_adv
                pg_loss = -torch.min(pg1, pg2).mean()

                v_loss_unclipped = (value - mb_returns).pow(2)
                v_clipped = mb_values_old + torch.clamp(value - mb_values_old, -self.cfg.clip_coef, self.cfg.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent = entropy.mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

    def train(self, total_updates: int = 50):
        for up in range(1, total_updates+1):
            obs_t, disc_t, cont_t, logp_old, values_old, returns, adv = self.collect_rollout()
            self.update(obs_t, disc_t, cont_t, logp_old, values_old, returns, adv)
            # мониторинг
            eq = self.env.info_last.get("equity", float("nan"))
            print(f"Update {up:03d}: last_equity={eq:,.2f}")

# =============== ДЕМО-ЗАПУСК ===============

def make_or_load_sample_csv(path: str, n_symbols: int = 4, bars: int = 6000):
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['timestamp'])
    rng = np.random.RandomState(0)
    base_ts = pd.date_range("2024-01-01", periods=bars, freq="15min", tz="UTC")
    rows = []
    for s in range(n_symbols):
        px = 100 + np.cumsum(rng.randn(bars)*0.2)
        px = np.maximum(px, 1.0)
        vol = rng.lognormal(mean=12, sigma=0.3, size=bars).astype(np.float64)
        for t in range(bars):
            close = px[t]
            open_ = close * (1 + rng.randn()*0.0005)
            high  = max(open_, close) * (1 + abs(rng.randn())*0.001)
            low   = min(open_, close) * (1 - abs(rng.randn())*0.001)
            rows.append({
                "timestamp": base_ts[t],
                "symbol": f"S{s+1}",
                "open": float(open_),
                "high": float(high),
                "low":  float(low),
                "close": float(close),
                "volume": float(vol[t]),
                # можно добавить spread_bp строкой ниже для правдоподобности:
                # "spread_bp": float(5 + abs(rng.randn()*2.0)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    DATA = "market_sample.csv"
    df = make_or_load_sample_csv(DATA)
    symbols = sorted(df['symbol'].unique().tolist())[:4]
    cfg_env = EnvConfig(window_size=64, initial_equity=1_000_000.0,
                        commission_bp=2.0, borrow_fee_bp_day=20.0,
                        spread_bp=8.0, impact_coef=2e-4, max_leverage=2.0,
                        feature_set="tech")

    env = MultiAssetTradingEnv(df, symbols=symbols, cfg=cfg_env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo = PPOAgent(env, PPOConfig(device=device, rollout_steps=1024, update_epochs=6, minibatch_size=128, lr=3e-4))
    ppo.train(total_updates=30)
