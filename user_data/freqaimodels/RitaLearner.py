import gc
import logging
import math
import time
from typing import Any, Dict, List, Optional, Type

import numpy as np

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class RitaLearner(ReinforcementLearner):
    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """Add memory management to training"""
        try:
            model = super().fit(data_dictionary, dk, **kwargs)
            gc.collect()  # Force garbage collection after training
            logger.info(f"Model trained and in use: {dk.data_path}")
            return model
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            gc.collect()  # Ensure memory is freed even on error
            raise

    class MyRLEnv(Base5ActionRLEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.win_streak = 0
            self.lose_streak = 0

            self.win_factor = self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
            self.designated_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)

            """
            Use this matrix to define the valid of the action bsed on the current position
            """
            # 0: Neutral, 1: Long Enter, 2: Long Eixt, 3: Short Enter, 4: Short Exit
            self.action_matrix = np.array(
                [
                    [+1, -1, -1, -1, +1],  # Short position
                    [+1, +1, -1, -1, -1],  # Long position
                    [-1, +1, -1, +1, -1],  # Neutral position
                ]
            )

        # Function to get the matrix value based on row and column Enums
        def get_matrix_value(self, position: Positions, action: Actions):
            row_index = int(position.value)  # Convert Enum value to index
            col_index = action.value
            return self.action_matrix[row_index, col_index]

        def step(self, action: int):
            """Step with streak updates only on trade completion"""
            # Get the current position before step
            previous_position = self._position

            # Execute main step logic
            observation, reward, done, truncated, info = super().step(action)

            # Update streaks only when a trade is closed
            if (
                previous_position in (Positions.Long, Positions.Short)
                and self._position == Positions.Neutral
            ):
                # A trade was just closed
                last_trade = self.trade_history[-1] if self.trade_history else None
                if last_trade:
                    profit = last_trade.get("profit", 0)
                    if profit > 0:
                        self.win_streak = min(self.win_streak + 1, 10)
                        self.lose_streak = 0
                    else:
                        self.lose_streak = min(self.lose_streak + 1, 10)
                        self.win_streak = 0

            return observation, reward, done, truncated, info

        def reset(self, seed=None):
            """Reset environment state including streaks"""

            # Reset streak counters
            self.win_streak = 0
            self.lose_streak = 0

            # Call base reset
            return super().reset(seed)

        def get_last_trade(self):
            if len(self.trade_history) > 0:
                return self.trade_history[-1]
            return None

        def _is_valid(self, action):
            """Determine if the action is valid"""
            return self.get_matrix_value(self._position, Actions(action)) == 1

        def calculate_reward(self, action: int) -> float:
            if not self._is_valid(action):
                return -1

            trade_duration = self._current_tick - (
                self._last_trade_tick if self._last_trade_tick is not None else self._current_tick
            )

            # For entering trades
            if action in (Actions.Long_enter.value, Actions.Short_enter.value):
                return 1

            # For exiting positions
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                return self.calculate_exit_reward(self.get_unrealized_profit(), trade_duration)

            # For neutral actions
            return self.calculate_neutral_reward(trade_duration)

        def calculate_neutral_reward(self, trade_duration):
            """
            Reward function for neutral actions
            """
            designated_trade_duration = self.designated_trade_duration
            return 1 - ((trade_duration / designated_trade_duration) ** 3)

        def calculate_exit_reward(self, unrealized_profit, trade_duration):
            """
            Reward function for winning trades
            """
            designated_trade_duration = self.designated_trade_duration

            result = (
                unrealized_profit / self.profit_aim
            ) ** 5  # Value Range: (-)infinite to (+)positive

            result *= 1 + ((self.win_streak + self.lose_streak) / 10) ** (3 / 20)
            result *= (
                1
                + (max((designated_trade_duration - trade_duration), 0) / designated_trade_duration)
                ** 2
            )
            return result
