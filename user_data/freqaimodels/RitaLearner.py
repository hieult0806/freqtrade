import gc
import logging
import math
from typing import Any, Dict, List, Optional, Type

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class RitaLearner(ReinforcementLearner):
    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        model = super().fit(data_dictionary, dk, **kwargs)
        gc.collect()
        logger.info(f"Model trained and in use: {dk.model_filename}")
        return model

    class MyRLEnv(Base5ActionRLEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.win_streak = 0
            self.lose_streak = 0

        def step(self, action: int):
            observation, reward, done, truncated, info = super().step(action)

            # Update win/lose streak based on reward
            if reward > 0:
                self.win_streak += 1
                self.lose_streak = 0
            elif reward < 0:
                self.lose_streak += 1
                self.win_streak = 0

            return observation, reward, done, truncated, info

        def reset(self, seed=None):
            self.win_streak = 0
            self.lose_streak = 0
            return super().reset(seed)

        def get_last_trade(self):
            if len(self.trade_history) > 0:
                return self.trade_history[-1]
            return None

        def calculate_reward(self, action: int) -> float:
            # Penalize if the action is not valid
            if not self._is_valid(action):
                return -2
            # time_have_no_trade = self.get_time_have_no_trade()
            win_factor = self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)

            # reward agent for entering trades
            if (
                action in (Actions.Long_enter.value, Actions.Short_enter.value)
                and self._position == Positions.Neutral
            ):
                exploit = 1
                last_trade = self.get_last_trade()
                current_price = self.current_price() * (1 + self.fee)
                if last_trade is not None:
                    if last_trade["type"] == "exit_long" and action == Actions.Long_enter:
                        exploit = last_trade["price"] / current_price
                    elif last_trade["type"] == "exit_short" and action == Actions.Short_enter:
                        exploit = current_price / last_trade["price"]
                return min(win_factor, win_factor * (exploit**2))

            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            trade_duration = self._current_tick - (
                self._last_trade_tick if self._last_trade_tick is not None else self._current_tick
            )

            # discourage sitting in position
            if (action != Actions.Long_exit.value and self._position == Positions.Long) or (
                action != Actions.Short_exit.value and self._position == Positions.Short
            ):
                return -((trade_duration * win_factor) / max_trade_duration)

            # reward agent when close a trade based on the profit
            if (action == Actions.Long_exit.value and self._position == Positions.Long) or (
                action == Actions.Short_exit.value and self._position == Positions.Short
            ):
                p = self.get_unrealized_profit()
                g = self.profit_aim
                m = max_trade_duration
                t = trade_duration
                w = win_factor
                s = self.win_streak
                l_o = self.lose_streak

                h = abs(self._total_profit)

                if p > 0:
                    return self.calculate_win_reward(p, g, m, t, w, h, s)
                else:
                    return self.calculate_loose_reward(p, g, m, t, w, h, l_o)

            return -(pow((trade_duration / max_trade_duration), 2) * pow(win_factor, 2))

        def calculate_win_reward(self, p, g, m, t, w, h, s):
            term1 = (p / g) + 1
            term2 = (m**2 * p + 1) / (t**2 + m)
            term3 = w
            term4 = (h + 1) ** 2
            term5 = s

            reward = (term1 * term2 * term3 * term4) + term5
            return reward

        def calculate_loose_reward(self, p, g, m, t, w, h, l_o):
            term1 = (p / g) + 1
            term2 = (abs(m - t) + 1) / m
            term3 = (h + 1) ** 2
            term4 = l_o

            reward = -abs(term1 * term2 * term3 + term4)
            return reward
