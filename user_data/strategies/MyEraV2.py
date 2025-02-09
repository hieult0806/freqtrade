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


class MyEraV2(IStrategy):
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
    stoploss = -0.035

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
        # Check for 5 consecutive green candles
        dataframe["green_candles"] = (
            (dataframe["close"] > dataframe["open"])
            & (dataframe["close"].shift(1) > dataframe["open"].shift(1))
            & (dataframe["close"].shift(2) > dataframe["open"].shift(2))
            & (dataframe["close"].shift(3) > dataframe["open"].shift(3))
            & (dataframe["close"].shift(4) > dataframe["open"].shift(4))
        )

        dataframe.loc[
            (dataframe["green_candles"] & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]
        ] = [1, "LONG"]

        # Check for 5 consecutive red candles
        dataframe["red_candles"] = (
            (dataframe["close"] < dataframe["open"])
            & (dataframe["close"].shift(1) < dataframe["open"].shift(1))
            & (dataframe["close"].shift(2) < dataframe["open"].shift(2))
            & (dataframe["close"].shift(3) < dataframe["open"].shift(3))
            & (dataframe["close"].shift(4) < dataframe["open"].shift(4))
        )

        dataframe.loc[
            (dataframe["red_candles"] & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]
        ] = [1, "SHORT"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # if len(dataframe) <= 10:
        #     return dataframe
        dataframe["price_2_candles_ago"] = dataframe["close"].shift(4)

        # Exit long if the price drops by 2% after 2 candles
        dataframe["drop_2_percent"] = (
            dataframe["price_2_candles_ago"] - dataframe["close"]
        ) / dataframe["price_2_candles_ago"] >= 0.03

        # Exit short if the price increases by 2% after 2 candles
        dataframe["rise_2_percent"] = (
            dataframe["close"] - dataframe["price_2_candles_ago"]
        ) / dataframe["price_2_candles_ago"] >= 0.03

        dataframe.loc[dataframe["drop_2_percent"], "exit_long"] = 1

        dataframe.loc[dataframe["rise_2_percent"], "exit_short"] = 1

        return dataframe

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
