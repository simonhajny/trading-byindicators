import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
import time
import requests
from scipy.optimize import differential_evolution
from collections import deque
import traceback


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    symbol: str = "ADAUSDT"
    initial_balance: float = 10000.0
    optimization_interval: int = 60  # Optimize every 60 minutes
    price_history_size: int = 1000  # Store 1000 recent price points
    log_interval: int = 1  # Log every minute


CONFIG = TradingConfig()


class PriceData:
    def __init__(self, max_size: int = CONFIG.price_history_size):
        self.prices = deque(maxlen=max_size)

    def update(self, price: float) -> None:
        self.prices.append(price)

    def get_series(self) -> pd.Series:
        return pd.Series(self.prices)


class TechnicalIndicator:
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> float:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean().iloc[-1]
        rs = gain / (loss + 1e-10)  # Adding epsilon to prevent division by zero
        return 100 - (100 / (1 + rs))


class SignalGenerator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def generate_signal(self, price_data: PriceData) -> int:
        if len(price_data.prices) < self.params['rsi_period']:
            return 0

        rsi = TechnicalIndicator.calculate_rsi(price_data.get_series(), period=self.params['rsi_period'])

        if rsi < self.params['rsi_oversold']:
            return 1
        elif rsi > self.params['rsi_overbought']:
            return -1
        return 0


class Position:
    def __init__(self, type: str, entry_price: float, leverage: float, size: float, initial_margin: float):
        self.type = type
        self.entry_price = entry_price
        self.leverage = leverage
        self.size = size
        self.initial_margin = initial_margin

    def calculate_pnl(self, current_price: float) -> float:
        price_change = (current_price - self.entry_price) / self.entry_price
        return self.size * price_change * current_price * (1 if self.type == 'long' else -1)


class RiskManager:
    @staticmethod
    def calculate_liquidation_threshold(leverage: float, position_type: str) -> float:
        if position_type == 'long':
            return 1 - (1 / leverage)
        elif position_type == 'short':
            return 1 + (1 / leverage)
        else:
            raise ValueError("Invalid position type. Must be 'long' or 'short'.")

    @staticmethod
    def check_liquidation(position: Position, current_price: float) -> bool:
        liquidation_threshold = RiskManager.calculate_liquidation_threshold(position.leverage, position.type)
        price_change = (current_price - position.entry_price) / position.entry_price
        
        if position.type == 'long':
            return price_change <= -liquidation_threshold
        elif position.type == 'short':
            return price_change >= liquidation_threshold


class PositionManager:
    def __init__(self, balance: float, params: Dict[str, Any]):
        self.balance = balance
        self.params = params
        self.position = None
        self.trades = []
        self.liquidations = 0

    def open_position(self, signal: int, price: float, timestamp: datetime) -> None:
        self._close_position(price, timestamp, 'new_signal')

        risk_amount = self.balance * self.params['risk_per_trade']
        leverage = self.params['leverage']
        position_size = risk_amount * leverage / price

        if 0 < position_size < np.inf:
            self.position = Position(
                type='long' if signal == 1 else 'short',
                entry_price=price,
                leverage=leverage,
                size=position_size,
                initial_margin=risk_amount
            )
            self._log_trade('open', price, timestamp)

    def _close_position(self, price: float, timestamp: datetime, reason: str) -> None:
        if self.position is None:
            return

        if reason == 'liquidation':
            profit_or_loss = -self.position.initial_margin
            self.liquidations += 1
        else:
            profit_or_loss = self.position.calculate_pnl(price)

        self.balance += profit_or_loss
        self._log_trade('close', price, timestamp, reason, profit_or_loss)
        self.position = None

    def _log_trade(self, trade_type: str, price: float, timestamp: datetime, reason: str = None, profit_loss: float = None) -> None:
        trade = {
            'type': trade_type,
            'price': f"{price:.3f}",  # Ensure price is displayed with 3 decimal places
            'balance': f"{self.balance:.2f}",
            'timestamp': timestamp
        }
        if trade_type == 'open':
            trade['position_type'] = self.position.type
        if trade_type == 'close':
            trade['reason'] = reason
            trade['profit_loss'] = f"{profit_loss:.2f}" if profit_loss is not None else None

        self.trades.append(trade)
        logger.info(f"{'Opened' if trade_type == 'open' else 'Closed'} {self.position.type} position at {price:.3f}. "
                    f"{'Reason: ' + reason + '.' if reason else ''} "
                    f"{'P/L: ' + f'{profit_loss:.2f}' + '.' if profit_loss is not None else ''} "
                    f"Balance: {self.balance:.2f}")

    def handle_position(self, price: float, signal: int, timestamp: datetime) -> None:
        if self.position is None:
            if signal != 0:
                self.open_position(signal, price, timestamp)
            return

        if RiskManager.check_liquidation(self.position, price):
            self._close_position(price, timestamp, 'liquidation')
            return

        if ((self.position.type == 'long' and price >= self.position.entry_price * (1 + self.params['target_percent'])) or
            (self.position.type == 'short' and price <= self.position.entry_price * (1 - self.params['target_percent']))):
            self._close_position(price, timestamp, 'take_profit')
            return

        if (signal == 1 and self.position.type == 'short') or (signal == -1 and self.position.type == 'long'):
            self.open_position(signal, price, timestamp)


class TradingStrategy:
    def __init__(self, params: Dict[str, Any], initial_balance: float = CONFIG.initial_balance):
        self.params = params
        self.price_data = PriceData()
        self.signal_generator = SignalGenerator(params)
        self.position_manager = PositionManager(initial_balance, params)
        self.last_optimization_time = datetime.now()

    def execute_trade(self, price: float, timestamp: datetime) -> None:
        self.price_data.update(price)
        signal = self.signal_generator.generate_signal(self.price_data)
        self.position_manager.handle_position(price, signal, timestamp)

    def update_params(self, new_params: Dict[str, Any]) -> None:
        self.params.update(new_params)
        self.signal_generator.params = self.params
        self.position_manager.params = self.params

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "current_price": f"{self.price_data.prices[-1]:.3f}" if self.price_data.prices else None,
            "current_rsi": f"{TechnicalIndicator.calculate_rsi(self.price_data.get_series(), self.params['rsi_period']):.2f}",
            "balance": f"{self.position_manager.balance:.2f}",
            "current_position": self.position_manager.position,
            "current_params": self.params
        }


def fetch_price(symbol: str) -> float:
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return round(float(data['price']), 3)
    except requests.RequestException as e:
        logger.error(f"Error fetching price: {str(e)}")
        raise


def objective_function(params: List[float], price_history: List[float], initial_balance: float) -> float:
    strategy = TradingStrategy({
        'rsi_period': int(params[0]),
        'rsi_oversold': params[1],
        'rsi_overbought': params[2],
        'leverage': params[3],
        'risk_per_trade': params[4],
        'target_percent': params[5],
    }, initial_balance)

    for price in price_history:
        strategy.execute_trade(price, datetime.now())
    return -strategy.position_manager.balance


def optimize_parameters(price_history: List[float], initial_balance: float) -> Dict[str, Any]:
    bounds = [
        (10, 20),    # rsi_period
        (15, 25),    # rsi_oversold
        (75, 85),    # rsi_overbought
        (10, 25),    # leverage
        (0.1, 0.5),  # risk_per_trade
        (0.01, 0.1), # target_percent
    ]
    result = differential_evolution(objective_function, bounds, args=(price_history, initial_balance), maxiter=10, popsize=10, updating='deferred', workers=-1)

    return {
        'rsi_period': int(result.x[0]),
        'rsi_oversold': result.x[1],
        'rsi_overbought': result.x[2],
        'leverage': result.x[3],
        'risk_per_trade': result.x[4],
        'target_percent': result.x[5],
    }


def main():
    initial_params = {
        'rsi_period': 14,
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'leverage': 15,
        'risk_per_trade': 0.1,
        'target_percent': 0.03,
    }

    trading_strategy = TradingStrategy(initial_params)
    logger.info("Starting real-time trading simulation with dynamic optimization...")
    logger.info(f"Initial parameters: {initial_params}")
    last_log_time = datetime.now()

    while True:
        try:
            current_price = fetch_price(CONFIG.symbol)
            current_time = datetime.now()
            trading_strategy.execute_trade(current_price, current_time)

            if (current_time - last_log_time).total_seconds() / 60 >= CONFIG.log_interval:
                state = trading_strategy.get_current_state()
                logger.info(f"Current state: Price={state['current_price']}, RSI={state['current_rsi']}, "
                            f"Balance={state['balance']}, Position={state['current_position']}")
                logger.info(f"Current parameters: {state['current_params']}")
                last_log_time = current_time

            if (current_time - trading_strategy.last_optimization_time).total_seconds() / 60 >= CONFIG.optimization_interval:
                logger.info("Starting parameter optimization...")
                price_history = list(trading_strategy.price_data.prices)
                optimized_params = optimize_parameters(price_history, trading_strategy.position_manager.balance)
                trading_strategy.update_params(optimized_params)
                trading_strategy.last_optimization_time = current_time
                logger.info(f"Optimization complete. New parameters: {optimized_params}")

            time.sleep(60)

        except requests.RequestException as e:
            logger.error(f"Error fetching price: {str(e)}")
            time.sleep(60)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            time.sleep(60)

if __name__ == "__main__":
    main()