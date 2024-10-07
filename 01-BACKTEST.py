import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import logging
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    price_data_path: str
    signal_data_path: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0

class DataLoader:
    @staticmethod
    def load_price_data(config: BacktestConfig) -> pd.DataFrame:
        try:
            df = pd.read_csv(config.price_data_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[(df.index >= config.start_date) & (df.index <= config.end_date)]
            return df[['close']]
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def load_signal_data(config: BacktestConfig) -> pd.DataFrame:
        try:
            df = pd.read_csv(config.signal_data_path, parse_dates=['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df = df[(df.index >= config.start_date) & (df.index <= config.end_date)]
            return df
        except Exception as e:
            logger.error(f"Error loading signal data: {str(e)}")
            return pd.DataFrame()

class SignalProcessor:
    @staticmethod
    def process_signals(signal_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        signal_types = signal_data['Signal_Name'].unique()
        processed_signals = {}
        for signal_type in signal_types:
            signal_df = signal_data[signal_data['Signal_Name'] == signal_type].copy()
            signal_df['Signal'] = signal_df['Signal_Type'].map({'LONG': 1, 'SHORT': -1}).fillna(0).astype(int)
            processed_signals[signal_type] = signal_df[['Signal', 'Price']]
        return processed_signals

class Position:
    def __init__(self, entry_price: float, size: float, direction: int):
        self.entry_price = entry_price
        self.size = size
        self.direction = direction  # 1 for long, -1 for short

    def calculate_pnl(self, current_price: float) -> float:
        return self.direction * self.size * (current_price - self.entry_price)

class RiskManager:
    def __init__(self, leverage: float, position_size: float, target_percent: float, stop_loss_percent: float):
        self.leverage = leverage
        self.position_size = position_size
        self.target_percent = target_percent
        self.stop_loss_percent = stop_loss_percent

    def calculate_position_size(self, account_balance: float, entry_price: float) -> float:
        return (account_balance * self.position_size * self.leverage) / entry_price

    def should_take_profit(self, position: Position, current_price: float) -> bool:
        return position.calculate_pnl(current_price) / (position.size * position.entry_price) >= self.target_percent

    def should_stop_loss(self, position: Position, current_price: float) -> bool:
        return position.calculate_pnl(current_price) / (position.size * position.entry_price) <= -self.stop_loss_percent

class TradeLogger:
    def __init__(self):
        self.trades = []

    def log_trade(self, timestamp, signal_type, action, price, size, pnl=None, balance=None, direction=None):
        self.trades.append({
            'timestamp': timestamp,
            'signal_type': signal_type,
            'action': action,
            'price': price,
            'size': size,
            'pnl': pnl,
            'balance': balance,
            'direction': direction
        })

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

class Backtester:
    def __init__(self, config: BacktestConfig, risk_params: Dict[str, float]):
        self.config = config
        self.risk_manager = RiskManager(**risk_params)
        self.trade_logger = TradeLogger()
        self.balance = config.initial_balance
        self.position = None
        self.risk_params = risk_params

    def run(self, price_data: pd.DataFrame, signals: Dict[str, pd.DataFrame]):
        for signal_type, signal_df in signals.items():
            logger.info(f"Running backtest for {signal_type} with parameters: {self.risk_params}")
            self.balance = self.config.initial_balance
            self.position = None
            
            merged_data = price_data.join(signal_df, how='left').copy()
            merged_data.loc[:, 'Signal'] = merged_data['Signal'].fillna(0)

            for timestamp, row in merged_data.iterrows():
                current_price = row['close']
                signal = row['Signal']

                if self.position:
                    should_close = (
                        self.risk_manager.should_take_profit(self.position, current_price) or
                        self.risk_manager.should_stop_loss(self.position, current_price) or
                        (signal != 0 and signal != self.position.direction)
                    )
                    if should_close:
                        pnl = self.position.calculate_pnl(current_price)
                        self.balance += pnl
                        self.trade_logger.log_trade(timestamp, signal_type, 'close', current_price, self.position.size, pnl, self.balance, 'long' if self.position.direction == 1 else 'short')
                        self.position = None

                if signal != 0 and not self.position:
                    size = self.risk_manager.calculate_position_size(self.balance, current_price)
                    self.position = Position(current_price, size, signal)
                    direction = 'long' if signal == 1 else 'short'
                    self.trade_logger.log_trade(timestamp, signal_type, 'open', current_price, size, None, self.balance, direction)

            if self.position:
                pnl = self.position.calculate_pnl(current_price)
                self.balance += pnl
                self.trade_logger.log_trade(timestamp, signal_type, 'close', current_price, self.position.size, pnl, self.balance, 'long' if self.position.direction == 1 else 'short')

            logger.info(f"Backtest completed for {signal_type}. Final balance: {self.balance:.8f}")

        return self.balance

    def get_results(self) -> pd.DataFrame:
        return self.trade_logger.get_trades_df()

class BayesianOptimizer:
    def __init__(self, config: BacktestConfig, price_data: pd.DataFrame, signals: Dict[str, pd.DataFrame]):
        self.config = config
        self.price_data = price_data
        self.signals = signals
        self.bounds = np.array([
            [20, 40],       # leverage
            [0.1, 0.5],     # position_size
            [0.004, 0.01],  # target_percent
            [0.005, 0.02]   # stop_loss_percent
        ])
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=25)

    def objective(self, params):
        leverage, position_size, target_percent, stop_loss_percent = params
        risk_params = {
            'leverage': leverage,
            'position_size': position_size,
            'target_percent': target_percent,
            'stop_loss_percent': stop_loss_percent
        }
        backtester = Backtester(self.config, risk_params)
        return -backtester.run(self.price_data, self.signals)

    def expected_improvement(self, X, X_sample, Y_sample, epsilon=1e-3):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(X_sample)

        mu_sample_opt = np.max(mu_sample)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - epsilon
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def optimize(self, n_iter=10):
        dim = self.bounds.shape[0]
        X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(5, dim))
        Y_sample = np.array([self.objective(params) for params in X_sample])

        X_sample = X_sample.reshape(-1, dim)
        Y_sample = Y_sample.reshape(-1, 1)

        start_time = time.time()
        for i in range(n_iter):
            elapsed_time = time.time() - start_time
            estimated_time_remaining = (elapsed_time / (i + 1)) * (n_iter - i - 1)
            sys.stdout.write(f"\rOptimization progress: {i+1}/{n_iter} | Est. time remaining: {estimated_time_remaining:.2f}s")
            sys.stdout.flush()

            self.gp.fit(X_sample, Y_sample)

            X = self.sample_next_hyperparameter()    
            ei = self.expected_improvement(X, X_sample, Y_sample)
            X_next = X[np.argmax(ei)]
            Y_next = self.objective(X_next)
 
            X_sample = np.vstack((X_sample, X_next.reshape(1, -1)))
            Y_sample = np.vstack((Y_sample, Y_next))

        sys.stdout.write("\n")
        return X_sample[np.argmin(Y_sample.reshape(-1))]

    def sample_next_hyperparameter(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(10000, self.bounds.shape[0]))

def save_results(optimized_params, backtest_results, file_path):
    all_results = []
    for signal_type, params in optimized_params.items():
        results = backtest_results[signal_type]
        results['signal_type'] = signal_type
        results['leverage'] = params['leverage']
        results['position_size'] = params['position_size']
        results['target_percent'] = params['target_percent']
        results['stop_loss_percent'] = params['stop_loss_percent']
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(file_path, index=False, float_format='%.8f')
    logger.info(f"All backtest results saved to {file_path}")

def optimize_parameters(config: BacktestConfig, price_data: pd.DataFrame, processed_signals: Dict[str, pd.DataFrame]):
    optimized_params = {}
    for signal_type, signals in processed_signals.items():
        print(f"\nOptimizing parameters for {signal_type}")
        optimizer = BayesianOptimizer(config, price_data, {signal_type: signals})
        optimal_params = optimizer.optimize()
        optimized_params[signal_type] = {
            'leverage': float(optimal_params[0]),
            'position_size': float(optimal_params[1]),
            'target_percent': float(optimal_params[2]),
            'stop_loss_percent': float(optimal_params[3])
        }
        print(f"\nOptimized parameters for {signal_type}:")
        for param, value in optimized_params[signal_type].items():
            print(f"  {param}: {value:.6f}")
    return optimized_params

def run_backtest(config: BacktestConfig, price_data: pd.DataFrame, processed_signals: Dict[str, pd.DataFrame], optimized_params: Dict[str, Dict[str, float]]):
    backtest_results = {}
    for signal_type, params in optimized_params.items():
        print(f"\nRunning backtest for {signal_type}")
        backtester = Backtester(config, params)
        final_balance = backtester.run(price_data, {signal_type: processed_signals[signal_type]})
        backtest_results[signal_type] = backtester.get_results()
        print(f"Backtest completed for {signal_type}. Final balance: {final_balance:.2f}")
    return backtest_results

def main():
    config = BacktestConfig(
        price_data_path="/Users/simonlavalle/Downloads/b-data/ADAUSDT_1min_data.csv",
        signal_data_path="/Users/simonlavalle/Downloads/trade_signals_1m.csv",
        start_date="2024-01-01",
        end_date="2024-10-04",
        initial_balance=10000.0
    )

    price_data = DataLoader.load_price_data(config)
    signal_data = DataLoader.load_signal_data(config)
    processed_signals = SignalProcessor.process_signals(signal_data)

    print("Starting parameter optimization (this may take a while)...")
    optimized_params = optimize_parameters(config, price_data, processed_signals)

    print("\nStarting backtest with optimized parameters")
    backtest_results = run_backtest(config, price_data, processed_signals, optimized_params)

    save_results(optimized_params, backtest_results, os.path.expanduser("~/Downloads/backtest_results_all.csv"))
    print(f"\nAll backtest results saved to {os.path.expanduser('~/Downloads/backtest_results_all.csv')}")

if __name__ == "__main__":
    main()
