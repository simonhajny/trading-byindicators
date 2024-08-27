import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from scipy.optimize import differential_evolution

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    file_path: str
    initial_balance: float = 10000.0
    start_date: str = "2024-07-10"
    end_date: str = "2024-08-04"
    liquidation_threshold: float = 0.80  # 80% of initial margin

CONFIG = TradingConfig(file_path="/Users/simonlavalle/Downloads/b-data/ADAUSDT_15min_data.csv")

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)  # Adding epsilon to prevent division by zero
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_volatility(data: pd.Series, window: int = 20) -> pd.Series:
        return data.pct_change().rolling(window=window).std()

class SignalGenerator:
    def __init__(self, data: pd.DataFrame, params: Dict[str, Any]):
        self.data = data
        self.params = params

    def generate_signals(self) -> pd.DataFrame:
        self.data['RSI'] = TechnicalIndicators.calculate_rsi(self.data['close'], period=self.params['rsi_period'])
        self.data['Volatility'] = TechnicalIndicators.calculate_volatility(self.data['close'], window=self.params['volatility_lookback'])
        
        for i in range(len(self.data)):
            rsi = self.data['RSI'].iloc[i]
            vol = self.data['Volatility'].iloc[i]
            if (rsi < self.params['rsi_oversold'] or rsi > self.params['rsi_overbought']) and vol <= self.params['volatility_lower']:
                logger.debug(f"Close to signal at {self.data.index[i]}: RSI={rsi:.2f}, Volatility={vol:.6f}")
        
        self.data['Signal'] = np.where(
            (self.data['RSI'] < self.params['rsi_oversold']) & (self.data['Volatility'] > self.params['volatility_lower']), 1,
            np.where((self.data['RSI'] > self.params['rsi_overbought']) & (self.data['Volatility'] > self.params['volatility_lower']), -1, 0)
        )
        return self.data

class RiskManager:
    def __init__(self, params: Dict[str, Any], liquidation_threshold: float):
        self.params = params
        self.liquidation_threshold = liquidation_threshold

    def check_liquidation(self, position: Dict[str, Any], price: float) -> bool:
        price_change = (price - position['entry_price']) / position['entry_price']
        unrealized_pnl = position['size'] * price_change * price * (1 if position['type'] == 'long' else -1)
        current_margin = position['initial_margin'] + unrealized_pnl
        return current_margin <= position['initial_margin'] * self.liquidation_threshold

class PositionManager:
    def __init__(self, balance: float, risk_manager: RiskManager):
        self.balance = balance
        self.risk_manager = risk_manager
        self.position = None
        self.trades = []
        self.liquidations = 0

    def open_position(self, signal: int, price: float, timestamp: datetime) -> None:
        risk_amount = self.balance * self.risk_manager.params['risk_per_trade']
        leverage = self.risk_manager.params['leverage']
        position_size = risk_amount * leverage / price

        if 0 < position_size < np.inf:
            self.position = {
                'type': 'long' if signal == 1 else 'short',
                'entry_price': price,
                'leverage': leverage,
                'size': position_size,
                'initial_margin': risk_amount
            }
            self.trades.append({
                'type': 'open',
                'price': price,
                'balance': round(self.balance, 2),
                'position_type': self.position['type'],
                'timestamp': timestamp
            })

    def close_position(self, price: float, timestamp: datetime, reason: str) -> None:
        if self.position is None:
            return

        price_change = (price - self.position['entry_price']) / self.position['entry_price']
        profit_or_loss = self.position['size'] * price_change * price * (1 if self.position['type'] == 'long' else -1)
        self.balance += profit_or_loss
        self.trades.append({
            'type': 'close',
            'price': price,
            'balance': round(self.balance, 2),
            'position_type': self.position['type'],
            'timestamp': timestamp,
            'reason': reason,
            'profit_loss': round(profit_or_loss, 2)
        })
        self.position = None

    def handle_existing_position(self, price: float, signal: int, timestamp: datetime) -> None:
        if self.position is None:
            return

        # Check for liquidation first
        if self.risk_manager.check_liquidation(self.position, price):
            self.close_position(price, timestamp, 'liquidation')
            self.liquidations += 1
            return

        # Check for take profit
        if abs((price - self.position['entry_price']) / self.position['entry_price']) >= self.risk_manager.params['target_percent']:
            self.close_position(price, timestamp, 'take_profit')
        # Check for signal reversal
        elif (self.position['type'] == 'long' and signal == -1) or \
             (self.position['type'] == 'short' and signal == 1):
            self.close_position(price, timestamp, 'signal_reversal')

class TradingStrategy:
    def __init__(self, data: pd.DataFrame, params: Dict[str, Any], initial_balance: float = CONFIG.initial_balance):
        self.data = data
        self.params = params
        self.signal_generator = SignalGenerator(data, params)
        self.risk_manager = RiskManager(params, CONFIG.liquidation_threshold)
        self.position_manager = PositionManager(initial_balance, self.risk_manager)

    def execute_trades(self) -> None:
        self.data = self.signal_generator.generate_signals()
        for i in range(1, len(self.data)):
            signal = self.data['Signal'].iloc[i]
            price = self.data['close'].iloc[i]
            timestamp = self.data.index[i]

            if self.position_manager.position is None and signal != 0:
                self.position_manager.open_position(signal, price, timestamp)
            else:
                self.position_manager.handle_existing_position(price, signal, timestamp)

        # Close any open position at the end of the period
        if self.position_manager.position:
            last_price = self.data['close'].iloc[-1]
            self.position_manager.close_position(last_price, self.data.index[-1], 'end_of_period')

def objective_function(params, data):
    strategy = TradingStrategy(data, {
        'rsi_period': int(params[0]),
        'rsi_oversold': params[1],
        'rsi_overbought': params[2],
        'leverage': params[3],
        'risk_per_trade': params[4],
        'target_percent': params[5],
        'volatility_lookback': int(params[6]),
        'volatility_lower': params[7]
    })
    strategy.execute_trades()
    return -strategy.position_manager.balance  # Negative because we want to maximize the balance

def optimize_parameters(data: pd.DataFrame) -> Dict[str, Any]:
    bounds = [
        (10, 20),    # rsi_period
        (15, 30),    # rsi_oversold
        (70, 85),    # rsi_overbought
        (10, 25),     # leverage
        (0.1, 0.5), # risk_per_trade
        (0.01, 0.1),  # target_percent
        (10, 30),    # volatility_lookback
        (0.01, 0.05)  # volatility_lower
    ]
    result = differential_evolution(objective_function, bounds, args=(data,), maxiter=50, popsize=15, tol=0.01, workers=-1)
    return {
        'rsi_period': int(result.x[0]),
        'rsi_oversold': result.x[1],
        'rsi_overbought': result.x[2],
        'leverage': result.x[3],
        'risk_per_trade': result.x[4],
        'target_percent': result.x[5],
        'volatility_lookback': int(result.x[6]),
        'volatility_lower': result.x[7]
    }

def load_data(config: TradingConfig) -> pd.DataFrame:
    try:
        df = pd.read_csv(config.file_path, parse_dates=['timestamp'], index_col='timestamp')
        df = df[(df.index >= config.start_date) & (df.index <= config.end_date)]
        return df[['close']]
    except FileNotFoundError:
        logger.error(f"Data file not found: {config.file_path}")
        return pd.DataFrame()

if __name__ == "__main__":
    data = load_data(CONFIG)
    if not data.empty:
        logger.info("Starting parameter optimization...")
        optimized_params = optimize_parameters(data)
        logger.info(f"Optimized Parameters: {optimized_params}")

        logger.info("Running backtest with optimized parameters...")
        trading_strategy = TradingStrategy(data, optimized_params)
        trading_strategy.execute_trades()

        # Print out final balance and other statistics
        logger.info(f"Final Balance: {trading_strategy.position_manager.balance}")
        logger.info(f"Number of Trades: {len(trading_strategy.position_manager.trades)}")
        logger.info(f"Number of Liquidations: {trading_strategy.position_manager.liquidations}")
        
        # Log only the first and last 5 trades for brevity
        for trade in trading_strategy.position_manager.trades[:5]:
            logger.info(trade)
        if len(trading_strategy.position_manager.trades) > 10:
            logger.info("...")
        for trade in trading_strategy.position_manager.trades[-5:]:
            logger.info(trade)