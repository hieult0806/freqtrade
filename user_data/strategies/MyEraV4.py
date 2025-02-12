# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class MyEraV4(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = True

    minimal_roi = {"500": 0.005, "400": 0.01, "200": 0.02, "100": 0.01, "0": 0.005}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.02

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 25

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    leverage_rate = 1
    conscutive_candles = 5

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "tema": {},
                "sar": {"color": "white"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                },
            },
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_green = dataframe["close"] > dataframe["open"]
        is_red = dataframe["close"] < dataframe["open"]

        # 5 consecutive green/red candles
        dataframe["green_candles"] = (
            is_green.rolling(window=self.conscutive_candles).sum() == self.conscutive_candles
        )
        dataframe["red_candles"] = (
            is_red.rolling(window=self.conscutive_candles).sum() == self.conscutive_candles
        )

        long_condition = dataframe["green_candles"] & (dataframe["volume"] > 0)
        short_condition = dataframe["red_candles"] & (dataframe["volume"] > 0)

        dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = [1, "LONG"]
        dataframe.loc[short_condition, ["enter_short", "enter_tag"]] = [1, "SHORT"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_green = dataframe["close"] > dataframe["open"]
        is_red = dataframe["close"] < dataframe["open"]

        consecutive_break = self.conscutive_candles - 1

        # 4 consecutive green/red candles for exit signals
        dataframe["f_exit_short"] = (
            is_green.rolling(window=consecutive_break).sum() == consecutive_break
        )
        dataframe["f_exit_long"] = (
            is_red.rolling(window=consecutive_break).sum() == consecutive_break
        )

        dataframe.loc[dataframe["f_exit_long"], "exit_long"] = 1
        dataframe.loc[dataframe["f_exit_short"], "exit_short"] = 1

        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_close = dataframe["close"].iloc[-1]
        offset = 0.005
        amplified = 1 + offset if side == "short" else 1 - offset
        return current_close * amplified

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        return self.leverage_rate
