import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mlbacktester import AssetInfo, BaseStrategy, Order


@dataclass
class EnsembleParams:
    sma_short: int
    sma_long: int
    rsi_len: int
    rsi_mode: str  # "trend" | "contrarian"
    rsi_th: float
    donchian_n: int
    scale_trend: float
    scale_breakout: float
    weights: Tuple[float, float, float]  # (w_trend, w_rsi, w_breakout)


class Strategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbols: List[str] = cfg["backtester_config"]["symbol"]

    # ------------------------------
    # Helpers
    # ------------------------------
    @staticmethod
    def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
        out = a / b.replace({0: np.nan})
        return out.fillna(fill)

    @staticmethod
    def _tanh_scaled(x: pd.Series, k: float) -> pd.Series:
        return np.tanh(x / max(k, 1e-9))

    # ------------------------------
    # Core API
    # ------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 年率ボラティリティ（既存方針）
        span = 24 * 7 * 4  # 4週間
        if "volatility" not in df.columns:
            df["volatility"] = np.nan
        for symbol in df.index.get_level_values("symbol").unique():
            symbol_df = df.xs(symbol, level="symbol")
            log_return = np.log(symbol_df["close"]).diff()
            rolling_std = log_return.rolling(window=span).std()
            annualized_vola_symbol = rolling_std * np.sqrt(365.25 * 24)
            annualized_vola_symbol.fillna(1, inplace=True)
            df.loc[(symbol_df.index, symbol), "volatility"] = annualized_vola_symbol.values

        # Donchian, ATR, SMA群（共通特徴）
        for symbol in df.index.get_level_values("symbol").unique():
            _df = df.loc[(slice(None), symbol), :].copy()

            for n in [20, 40, 60, 80]:
                _df[f"sma_{n}"] = _df["close"].rolling(n, min_periods=1).mean()

            for n in [14, 21]:
                # 簡易ATR（TrueRangeのEMA等は使わずrolling）
                tr = (_df["high"] - _df["low"]).abs()
                _df[f"atr_{n}"] = tr.rolling(n, min_periods=1).mean()

            for n in [20, 40]:
                _df[f"donch_hi_{n}"] = _df["high"].rolling(n, min_periods=1).max()
                _df[f"donch_lo_{n}"] = _df["low"].rolling(n, min_periods=1).min()
                _df[f"donch_mid_{n}"] = (_df[f"donch_hi_{n}"] + _df[f"donch_lo_{n}"]) / 2.0

            df.loc[(slice(None), symbol), _df.columns] = _df

        return df

    def get_model(self, train_df: pd.DataFrame) -> Dict[str, EnsembleParams]:
        # 小さなグリッドで探索
        sma_short_cand = [8, 12, 16]
        sma_long_cand = [40, 60, 80]
        rsi_len_cand = [7, 14, 21]
        rsi_mode_cand = ["trend", "contrarian"]
        rsi_th_cand = [0.45, 0.5]
        donchian_n_cand = [20, 40]
        scale_trend_cand = [1.0, 1.5]
        scale_breakout_cand = [1.0, 1.5]
        weight_cand = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.33, 0.33, 0.34)]

        best: Dict[str, EnsembleParams] = {}

        for symbol in self.symbols:
            sym_df = train_df.loc[(slice(None), symbol), :].copy()
            sym_df = sym_df.sort_index()

            best_sr = -1e18
            best_params = None

            # できるだけ軽くするため、候補をサブサンプル
            for sma_s in sma_short_cand:
                for sma_l in sma_long_cand:
                    if sma_s >= sma_l:
                        continue
                    for rsi_len in rsi_len_cand:
                        for rsi_mode in rsi_mode_cand:
                            for rsi_th in rsi_th_cand:
                                for dn in donchian_n_cand:
                                    for sc_t in scale_trend_cand:
                                        for sc_b in scale_breakout_cand:
                                            for w in weight_cand:
                                                sr = self._score_params(sym_df, sma_s, sma_l, rsi_len, rsi_mode, rsi_th, dn, sc_t, sc_b, w)
                                                if sr > best_sr:
                                                    best_sr = sr
                                                    best_params = EnsembleParams(
                                                        sma_short=sma_s,
                                                        sma_long=sma_l,
                                                        rsi_len=rsi_len,
                                                        rsi_mode=rsi_mode,
                                                        rsi_th=rsi_th,
                                                        donchian_n=dn,
                                                        scale_trend=sc_t,
                                                        scale_breakout=sc_b,
                                                        weights=w,
                                                    )

            assert best_params is not None
            best[symbol] = best_params

        return best  # pickle保存可能

    def _score_params(
        self,
        df: pd.DataFrame,
        sma_short: int,
        sma_long: int,
        rsi_len: int,
        rsi_mode: str,
        rsi_th: float,
        donchian_n: int,
        scale_trend: float,
        scale_breakout: float,
        weights: Tuple[float, float, float],
    ) -> float:
        z_trend = self._trend_signal(df, sma_short, sma_long, scale_trend)
        z_rsi = self._rsi_signal(df, rsi_len, rsi_mode, rsi_th)
        z_bo = self._breakout_signal(df, donchian_n, scale_breakout)
        s = weights[0] * z_trend + weights[1] * z_rsi + weights[2] * z_bo
        s = s.clip(-1.0, 1.0)
        ret = df["close"].pct_change().fillna(0.0)
        strat = ret * s.shift(1).fillna(0.0)
        mu = float(strat.mean())
        sd = float(strat.std(ddof=0))
        return mu / sd if sd > 0 else -1e18

    def _trend_signal(self, df: pd.DataFrame, sma_short: int, sma_long: int, scale: float) -> pd.Series:
        s_short = df[f"sma_{sma_short}"]
        s_long = df[f"sma_{sma_long}"]
        z = self._safe_div(s_short - s_long, df["atr_14"].replace(0, np.nan).fillna(df["close"].rolling(14).std()))
        return self._tanh_scaled(z, scale)

    def _rsi_signal(self, df: pd.DataFrame, rsi_len: int, mode: str, th: float) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_len, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_len, min_periods=1).mean()
        rs = self._safe_div(avg_gain, avg_loss.replace(0, np.nan), fill=0.0)
        rsi = 100 - (100 / (1 + rs))
        sig = rsi / 50.0 - 1.0
        if mode == "contrarian":
            sig = sig.apply(lambda x: -x if abs(x) >= th else x)
        return sig

    def _breakout_signal(self, df: pd.DataFrame, n: int, scale: float) -> pd.Series:
        mid = df[f"donch_mid_{n}"]
        width = (df[f"donch_hi_{n}"] - df[f"donch_lo_{n}"]) / 2.0
        z = self._safe_div(df["close"] - mid, width.replace(0, np.nan), fill=0.0)
        return self._tanh_scaled(z, scale)

    def get_signal(self, preprocessed_df: pd.DataFrame, models: Dict[str, EnsembleParams]) -> pd.DataFrame:
        df = preprocessed_df.copy()
        frames: List[pd.DataFrame] = []
        for symbol in self.symbols:
            p = models[symbol]
            _df = df.loc[(slice(None), symbol), :].copy()
            z_trend = self._trend_signal(_df, p.sma_short, p.sma_long, p.scale_trend)
            z_rsi = self._rsi_signal(_df, p.rsi_len, p.rsi_mode, p.rsi_th)
            z_bo = self._breakout_signal(_df, p.donchian_n, p.scale_breakout)
            s = p.weights[0] * z_trend + p.weights[1] * z_rsi + p.weights[2] * z_bo
            _df["signal"] = s.clip(-1.0, 1.0)
            frames.append(_df)
        out = pd.concat(frames)
        out = out.sort_index(level=1).sort_index(level=0)
        return out

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info: AssetInfo):
        order_lst: list[Order] = []
        d = 0.35
        size_ratio = {"BTCUSDT": 0.1, "ETHUSDT": 1.5, "XRPUSDT": 4000}

        volatilities = {
            symbol: latest_signal.loc[(slice(None), symbol), :].iloc[0]["volatility"]
            for symbol in self.cfg["backtester_config"]["symbol"]
        }
        total_inv_vol = sum(1 / vol for vol in volatilities.values())
        risk_weights = {symbol: (1 / vol) / total_inv_vol for symbol, vol in volatilities.items()}

        for symbol in self.cfg["backtester_config"]["symbol"]:
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]
            pos_size = asset_info.signed_pos_sizes[symbol]
            signal_value = latest_signal_symbol.get("signal", 0.0)
            if pd.isna(signal_value):
                signal_value = 0.0
            target_position_size = (
                math.floor(signal_value / d) * 0.5 if signal_value > 0 else math.ceil(signal_value / d) * 0.5
            )

            match target_position_size:
                case 1:
                    annualized_risk_target = 0.5
                case 0.5:
                    annualized_risk_target = 0.25
                case -0.5:
                    annualized_risk_target = -0.25
                case -1:
                    annualized_risk_target = -0.5
                case _:
                    annualized_risk_target = 0

            relevant_vola = float(latest_signal_symbol["volatility"]) or 1.0
            target_size = (annualized_risk_target / relevant_vola) * size_ratio[symbol] * risk_weights[symbol]
            order_size = target_size - pos_size
            if order_size == 0:
                continue
            side = "BUY" if order_size > 0 else "SELL"

            min_lot = self.cfg["exchange_config"].get(symbol, {}).get("min_lot", 0)
            if abs(order_size) >= min_lot:
                order_lst.append(
                    Order(type="MARKET", side=side, size=abs(order_size), price=None, symbol=symbol)
                )

        return order_lst


