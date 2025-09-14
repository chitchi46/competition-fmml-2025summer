import math
from typing import Any, List

import numpy as np
import pandas as pd

from mlbacktester import AssetInfo, BaseStrategy, Order


class Strategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbols: List[str] = cfg["backtester_config"]["symbol"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        span = 24 * 7 * 4  # 4週間
        df = df.copy()
        df["volatility"] = np.nan

        for symbol in df.index.get_level_values("symbol").unique():
            symbol_df = df.xs(symbol, level="symbol")
            log_return = np.log(symbol_df["close"]).diff()
            rolling_std = log_return.rolling(window=span).std()
            annualized_vola_symbol = rolling_std * np.sqrt(365.25 * 24)
            annualized_vola_symbol.fillna(1, inplace=True)
            df.loc[(symbol_df.index, symbol), "volatility"] = annualized_vola_symbol.values

        return df

    def get_model(self, train_df: pd.DataFrame) -> list:
        windows = [9, 14, 18]
        models: list[int] = []

        for symbol in self.symbols:
            best_window = 1
            _df = train_df.loc[(slice(None), symbol), :].copy()
            best_return = -np.inf

            _df["delta"] = _df["close"].diff()
            _df["gain"] = _df["delta"].clip(lower=0)
            _df["loss"] = _df["delta"].clip(upper=0)

            for window in windows:
                _df["avg_gain"] = _df["gain"].rolling(window=window, min_periods=1).mean()
                _df["avg_loss"] = -_df["loss"].rolling(window=window, min_periods=1).mean()
                _df["RS"] = _df["avg_gain"] / _df["avg_loss"]
                _df["RSI"] = 100 - (100 / (1 + _df["RS"]))
                _df["signal"] = _df["RSI"] / 50 - 1

                _df["daily_return"] = _df["close"].pct_change()
                _df["strategy_return"] = _df["daily_return"] * _df["signal"].shift(1)
                _df["strategy_return"] = _df["strategy_return"].fillna(0)

                cumulative_return = float(_df["strategy_return"].sum())
                if cumulative_return > best_return:
                    best_return = cumulative_return
                    best_window = window

            models.append(best_window)

        return models

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list) -> pd.DataFrame:
        df = preprocessed_df.copy()
        frames: list[pd.DataFrame] = []

        for symbol, window in zip(self.symbols, models):
            _df = df.loc[(slice(None), symbol), :].copy()
            _df["delta"] = _df["close"].diff()
            _df["gain"] = _df["delta"].clip(lower=0)
            _df["loss"] = -_df["delta"].clip(upper=0)
            _df["avg_gain"] = _df["gain"].rolling(window=window, min_periods=1).mean()
            _df["avg_loss"] = _df["loss"].rolling(window=window, min_periods=1).mean()
            _df["RS"] = _df["avg_gain"] / _df["avg_loss"]
            _df["RSI"] = 100 - (100 / (1 + _df["RS"]))
            _df["signal"] = _df["RSI"] / 50 - 1
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


