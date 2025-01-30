import gc
import logging
import math
import time
import numpy as np
from typing import Any, Dict, List, Optional, Type

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
            logger.info(f"Model trained and in use: {dk.model_filename}")
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
            
            self.hold_threshold = 0.005  # 0.5% profit threshold for hold reward
            self.max_hold_reward = 2.0   # Maximum reward multiplier for holding
            self.loss_hold_penalty = 2.0  # Penalty multiplier for holding losing positions
            self.target_profit = 0.02    # 2% target profit
            self.profit_decay_rate = 0.5 # How quickly reward decreases after target
            self.win_factor = self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
            
            # Increase cleanup interval significantly to reduce overhead
            self._steps_since_cleanup = 0
            self._cleanup_interval = 100000  # Increased from 1000
            self._last_cleanup_time = time.time()
            self._min_cleanup_interval_seconds = 60 * 30  # Minimum 60 seconds between cleanups
            
        def should_cleanup(self) -> bool:
            """Determine if cleanup should be performed"""
            current_time = time.time()
            time_since_last_cleanup = current_time - self._last_cleanup_time
            
            # Always allow force cleanup
            if self._steps_since_cleanup >= self._cleanup_interval:
                logger.info(f"Cleanup triggered by step count: {self._steps_since_cleanup}")
                return True
                
            # Check if enough time has passed since last cleanup
            if time_since_last_cleanup < self._min_cleanup_interval_seconds:
                return False
                
            # Check if histories have grown significantly
            position_history_growth = len(self._position_history) - self._last_position_history_size
            trade_history_growth = len(self.trade_history) - self._last_trade_history_size
            
            if position_history_growth > self.window_size * 3:
                logger.info(f"Cleanup triggered by position history growth: {position_history_growth}")
                return True
                
            if trade_history_growth > self.window_size * 2:
                logger.info(f"Cleanup triggered by trade history growth: {trade_history_growth}")
                return True
                
            return False   
        
        def perform_cleanup(self):
            """Perform cleanup of environment state variables"""
            try:
                if not self.should_cleanup():
                    return
                    
                cleanup_start = time.time()
                logger.info("Starting environment cleanup...")
                
                # Track initial sizes
                initial_position_size = len(self._position_history)
                initial_trade_size = len(self.trade_history)
                
                # Clean position history
                if len(self._position_history) > self.window_size * 3:
                    start_positions = self._position_history[:self.window_size]
                    recent_positions = self._position_history[-self.window_size * 2:]
                    self._position_history = start_positions + recent_positions
                
                # Clean trade history
                if len(self.trade_history) > self.window_size * 2:
                    profitable_trades = [
                        trade for trade in self.trade_history 
                        if trade.get('profit', 0) > self.hold_threshold
                    ][-self.window_size:]  # Keep only most recent profitable trades
                    recent_trades = self.trade_history[-self.window_size:]
                    self.trade_history = list({
                        trade['index']: trade 
                        for trade in (profitable_trades + recent_trades)
                    }.values())
                
                # Clean up close trade profits if it exists
                if hasattr(self, 'close_trade_profit') and len(self.close_trade_profit) > self.window_size * 2:
                    self.close_trade_profit = self.close_trade_profit[-self.window_size:]
                
                # Clean history dict
                if self.history:
                    history_length = len(next(iter(self.history.values())))
                    if history_length > self.window_size * 2:
                        for key in self.history:
                            self.history[key] = self.history[key][-self.window_size * 2:]
                
                # Force garbage collection
                gc.collect()
                
                # Update tracking variables
                self._steps_since_cleanup = 0
                self._last_cleanup_time = time.time()
                self._last_position_history_size = len(self._position_history)
                self._last_trade_history_size = len(self.trade_history)
                
                # Log cleanup results
                cleanup_duration = time.time() - cleanup_start
                logger.info(
                    f"Cleanup completed in {cleanup_duration:.2f}s. "
                    f"Position history: {initial_position_size}->{len(self._position_history)}, "
                    f"Trade history: {initial_trade_size}->{len(self.trade_history)}"
                )
                    
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

        def step(self, action: int):
            """Step with streak updates only on trade completion"""
            try:
                self._steps_since_cleanup += 1
                self.perform_cleanup()
                
                # Get the current position before step
                previous_position = self._position
                
                # Execute main step logic
                observation, reward, done, truncated, info = super().step(action)
                
                # Update streaks only when a trade is closed
                if (previous_position in (Positions.Long, Positions.Short) and 
                    self._position == Positions.Neutral):
                    # A trade was just closed
                    last_trade = self.trade_history[-1] if self.trade_history else None
                    if last_trade:
                        profit = last_trade.get('profit', 0)
                        if profit > 0:
                            self.win_streak = min(self.win_streak + 1, 10)
                            self.lose_streak = 0
                        else:
                            self.lose_streak = min(self.lose_streak + 1, 10)
                            self.win_streak = 0
                
                return observation, reward, done, truncated, info
                
            except Exception as e:
                logger.error(f"Error in step: {str(e)}")
                self._steps_since_cleanup = self._cleanup_interval
                self.perform_cleanup()
                return self._get_observation(), -1.0, True, False, {}
        
        def calculate_hold_reward(self, unrealized_profit, trade_duration, max_duration):
            """Calculate reward for holding a position"""
            # If position is losing money, apply increasing penalty
            if unrealized_profit < 0:
                loss_size = abs(unrealized_profit)
                time_factor = trade_duration / max_duration
                # Penalty increases with both loss size and hold duration
                penalty = self.loss_hold_penalty * loss_size * (1 + time_factor)
                return -penalty
            
            # Calculate reward based on profit target
            if unrealized_profit >= self.target_profit:
                # Calculate how long we've been above target
                excess_profit = unrealized_profit - self.target_profit
                time_since_target = trade_duration / max_duration
                
                # Reward decays exponentially after hitting target
                decay_factor = math.exp(-self.profit_decay_rate * time_since_target)
                target_reward = self.max_hold_reward * decay_factor
                
                # Add small bonus for excess profit
                excess_reward = min(excess_profit, self.max_hold_reward * 0.5)
                
                return target_reward + excess_reward
            
            # If below target but above threshold, give proportional reward
            elif unrealized_profit >= self.hold_threshold:
                profit_ratio = unrealized_profit / self.target_profit
                time_factor = 1 - (trade_duration / max_duration)
                base_reward = self.max_hold_reward * profit_ratio * time_factor
                return max(base_reward, 0.1)
            
            # Below threshold - small penalty
            return -((trade_duration * self.win_factor) / max_duration)

        def reset(self, seed=None):
            """Reset environment state including streaks"""
            # Force cleanup
            self._steps_since_cleanup = self._cleanup_interval
            self.perform_cleanup()
            
            # Reset streak counters
            self.win_streak = 0
            self.lose_streak = 0
            
            # Call base reset
            return super().reset(seed)

        def get_last_trade(self):
            if len(self.trade_history) > 0:
                return self.trade_history[-1]
            return None

        def calculate_reward(self, action: int) -> float:
            if not self._is_valid(action):
                return -2

            win_factor = self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            trade_duration = self._current_tick - (
                self._last_trade_tick if self._last_trade_tick is not None else self._current_tick
            )
            
            # For entering trades
            if (action in (Actions.Long_enter.value, Actions.Short_enter.value) 
                and self._position == Positions.Neutral):
                exploit = 1
                last_trade = self.get_last_trade()
                current_price = self.current_price() * (1 + self.fee)
                if last_trade is not None:
                    if last_trade["type"] == "exit_long" and action == Actions.Long_enter:
                        exploit = last_trade["price"] / current_price
                    elif last_trade["type"] == "exit_short" and action == Actions.Short_enter:
                        exploit = current_price / last_trade["price"]
                return min(win_factor, win_factor * (exploit**2))

            # For neutral position
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            # For holding positions
            if ((action != Actions.Long_exit.value and self._position == Positions.Long) or
                (action != Actions.Short_exit.value and self._position == Positions.Short)):
                unrealized_profit = self.get_unrealized_profit()
                return self.calculate_hold_reward(unrealized_profit, trade_duration, max_trade_duration)

            # For exiting positions
            if ((action == Actions.Long_exit.value and self._position == Positions.Long) or
                (action == Actions.Short_exit.value and self._position == Positions.Short)):
                p = self.get_unrealized_profit()
                g = self.profit_aim
                m = max_trade_duration
                t = trade_duration
                w = win_factor
                s = self.win_streak
                l_o = self.lose_streak
                h = abs(self._total_profit)

                # Modified exit reward to consider target profit
                if p >= self.target_profit:
                    # Bonus for exiting after reaching target
                    target_bonus = 1.5
                    return self.calculate_win_reward(p, g, m, t, w, h, s) * target_bonus
                elif p > 0:
                    return self.calculate_win_reward(p, g, m, t, w, h, s)
                else:
                    return self.calculate_loose_reward(p, g, m, t, w, h, l_o)

            return -(pow((trade_duration / max_trade_duration), 2) * pow(win_factor, 2))

        def calculate_win_reward(self, p, g, m, t, w, h, s):
            term1 = (p / g) + 1           # Profit relative to profit aim
            term2 = (m**2 * p + 1) / (t**2 + m)  # Time efficiency factor
            term3 = w                     # Win factor
            term4 = (h + 1)**2             # Total profit impact
            term5 = s                     # Win streak bonus
            return (term1 * term2 * term3 * term4) + term5

        def calculate_loose_reward(self, p, g, m, t, w, h, l_o):
            term1 = (p / g) + 1          # Loss relative to profit aim
            term2 = (abs(m - t) + 1) / m    # Time factor
            term3 = (h + 1)**2             # Total profit impact
            term4 = l_o                  # Lose streak penalty
            return -abs(term1 * term2 * term3 + term4)
