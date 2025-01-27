# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
import math
from pandas import DataFrame, Series
import pandas_ta as pta
from typing import Optional, Union
from datetime import datetime, timedelta
from freqtrade.enums import NO_ECHO_MESSAGES, RPCMessageType
import logging
from functools import reduce
from freqtrade.persistence import Trade, Order
from typing import List, Optional, Dict

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RitaTrader(IStrategy):
    INTERFACE_VERSION = 3

    stoploss = -1
    trailing_stop = False
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True
    # Run "populate_indicators" only for new candle.
    process_only_new_candles = True
    can_short = True

    startup_candle_count = 60

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    def __init__(self, config) -> None:
        super().__init__(config)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        macd = ta.MACD(dataframe, fastperiod=6, slowperiod=12, signalperiod=5)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        dataframe["%-tema"] = ta.TEMA(dataframe, timeperiod=period)
        dataframe["%-adx"] = ta.ADX(dataframe)
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["%-bob"] = ta.BOP(dataframe)
        dataframe["%-atr"] = ta.ATR(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["%-bb_lowerband"] = bollinger["lower"]
        dataframe["%-bb_middleband"] = bollinger["mid"]
        dataframe["%-bb_upperband"] = bollinger["upper"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        dataframe["%-cci"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-trix"] = ta.TRIX(dataframe, timeperiod=period)
        dataframe["%-mom"] = ta.MOM(dataframe, timeperiod=period)
        dataframe["%-roc"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-willr"] = ta.WILLR(dataframe, timeperiod=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(dataframe["close"], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe[f"%-atr-period"] / dataframe["close"] * 1000
        dataframe["%-cmf-period"] = self.chaikin_mf(dataframe, periods=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(dataframe["close"], length=period)

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["hour_of_day"] = dataframe["date"].dt.hour

        dataframe["day_of_week_norm"] = (
            2 * math.pi * dataframe["day_of_week"] / dataframe["day_of_week"].max()
        )
        dataframe["hour_of_day_norm"] = (
            2 * math.pi * dataframe["hour_of_day"] / dataframe["hour_of_day"].max()
        )

        dataframe["%%-day_of_week_cos"] = np.cos(dataframe["day_of_week_norm"])
        dataframe["%%-hour_of_day_cos"] = np.cos(dataframe["hour_of_day_norm"])
        dataframe["%%-day_of_week_sin"] = np.sin(dataframe["day_of_week_norm"])
        dataframe["%%-hour_of_day_sin"] = np.sin(dataframe["hour_of_day_norm"])

        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        # For RL, there are no direct targets to set. This is filler (neutral)
        # until the agent sends an action.
        dataframe["&-action"] = 0
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [df["do_predict"] == 1, df["&-action"] == 1]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [df["do_predict"] == 1, df["&-action"] == 3]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-action"] == 2]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-action"] == 4]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if "short" in trade.enter_tag:
                num_shorts += 1
            elif "long" in trade.enter_tag:
                num_longs += 1

        if side == "long" and num_longs >= 5:
            return False

        if side == "short" and num_shorts >= 5:
            return False

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True

    def chaikin_mf(self, df, periods=20):
        close = df["close"]
        low = df["low"]
        high = df["high"]
        volume = df["volume"]
        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0)
        mfv *= volume
        cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
        return Series(cmf, name="cmf")

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: Optional[float],
        **kwargs,
    ) -> float:
        return self.config["leverage"]["tier"]
