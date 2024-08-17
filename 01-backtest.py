import pandas as pd
import talib
import itertools
import random
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
import os
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import KFold

# '/Users/simonlavalle/Downloads/b-data/ADAUSDT_1min_data.csv',  '/Users/simonlavalle/Downloads/b-data/ADAUSDT_1h_data.csv'

file_paths = ['/Users/simonlavalle/Downloads/b-data/ADAUSDT_1min_data.csv']
INITIAL_BALANCE = 10000

def fetch_historical_data(file_path: str, start_str: str, end_str: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data = data.loc[start_str:end_str]
        data[['open', 'high', 'low', 'close', 'volume', 'rsi']] = data[['open', 'high', 'low', 'close', 'volume', 'rsi']].astype(float)
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = data.copy()
    data['rsi'] = talib.RSI(data['close'], timeperiod=window)
    return data

def calculate_buy_and_hold_return(data: pd.DataFrame) -> float:
    start_price = data['close'].iloc[0]
    end_price = data['close'].iloc[-1]
    return (end_price - start_price) / start_price

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
    data = data.copy()
    data['middle_band'] = data['close'].rolling(window=window).mean()
    data['upper_band'] = data['middle_band'] + num_std_dev * data['close'].rolling(window=window).std()
    data['lower_band'] = data['middle_band'] - num_std_dev * data['close'].rolling(window=window).std()
    return data

class TradingStrategy:
    def __init__(self, target_percent: float, interval_step: int, rsi_oversold: float, rsi_overbought: float, leverage: float, risk_per_trade: float, bollinger_window: int = 20, num_std_dev: int = 2, scale_factor: float = 0.5, partial_take_profit: float = 0.5):
        self.target_percent = target_percent
        self.interval_step = interval_step
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.balance = INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.portfolio_values = []
        self.position_type = None
        self.last_trade_index = -interval_step
        self.bollinger_window = bollinger_window
        self.num_std_dev = num_std_dev
        self.daily_returns = []
        self.scale_factor = scale_factor  # New: Percentage of position to enter at each step
        self.partial_take_profit = partial_take_profit  # New: Percentage of position to take profit on

    def open_position(self, price: float, position_type: str, current_rsi: float, index: int):
        increment_position = (self.balance * self.leverage * self.risk_per_trade * self.scale_factor) / price
        self.position += increment_position
        self.entry_price = (self.entry_price * (self.position - increment_position) + price * increment_position) / self.position
        self.position_type = position_type
        self.last_trade_index = index

    def close_position(self, price: float, profit: float, index: int, partial=False):
        if partial:
            self.balance += profit * self.partial_take_profit
            self.position *= (1 - self.partial_take_profit)  # Reduce position size by the partial take profit factor
        else:
            self.balance += profit
            self.position = 0  # Close entire position

    def liquidate_position(self, price: float, loss: float, index: int):
        self.balance -= loss
        self.position = 0

    def calculate_sharpe_ratio(self) -> float:
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = avg_return / return_std if return_std != 0 else 0
        return sharpe_ratio

    def execute(self, data: pd.DataFrame) -> pd.Series:
        data = calculate_bollinger_bands(data, window=self.bollinger_window, num_std_dev=self.num_std_dev)
        prev_rsi = data['rsi'].iloc[0]

        for current_index, (index, row) in enumerate(data.iterrows()):
            current_rsi = row['rsi']
            if self.position == 0 and current_index >= self.last_trade_index + self.interval_step:
                if current_rsi <= self.rsi_oversold and prev_rsi > self.rsi_oversold and row['close'] <= row['lower_band']:
                    self.open_position(row['close'], 'long', current_rsi, current_index)
                elif current_rsi >= self.rsi_overbought and prev_rsi < self.rsi_overbought and row['close'] >= row['upper_band']:
                    self.open_position(row['close'], 'short', current_rsi, current_index)

            elif self.position != 0:
                if self.position_type == 'long':
                    if row['close'] >= self.entry_price * (1 + self.target_percent):
                        profit = self.position * (row['close'] - self.entry_price)
                        self.close_position(row['close'], profit, current_index)
                    elif row['close'] <= self.entry_price * (1 - (1 / self.leverage)):
                        loss = self.position * (self.entry_price - row['close'])
                        self.liquidate_position(row['close'], loss, current_index)
                    elif row['close'] >= self.entry_price * (1 + self.partial_take_profit * self.target_percent):
                        # Partial profit-taking
                        profit = self.position * self.partial_take_profit * (row['close'] - self.entry_price)
                        self.close_position(row['close'], profit, current_index, partial=True)

                elif self.position_type == 'short':
                    if row['close'] <= self.entry_price * (1 - self.target_percent):
                        profit = self.position * (self.entry_price - row['close'])
                        self.close_position(row['close'], profit, current_index)
                    elif row['close'] >= self.entry_price * (1 + (1 / self.leverage)):
                        loss = self.position * (row['close'] - self.entry_price)
                        self.liquidate_position(row['close'], loss, current_index)
                    elif row['close'] <= self.entry_price * (1 - self.partial_take_profit * self.target_percent):
                        # Partial profit-taking
                        profit = self.position * self.partial_take_profit * (self.entry_price - row['close'])
                        self.close_position(row['close'], profit, current_index, partial=True)

            self.portfolio_values.append(self.balance)
            prev_rsi = current_rsi

        data['portfolio_value'] = self.portfolio_values
        self.daily_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        return data['portfolio_value']
    

def k_fold_cross_validation(file_paths: List[str], start_date: str, end_date: str, params: Dict[str, Any], k: int = 5) -> float:
    kf = KFold(n_splits=k)
    sharpe_ratios = []

    for file_path in file_paths:
        data = fetch_historical_data(file_path, start_date, end_date)
        data = calculate_rsi(data, window=params['rsi_window'])

        for train_index, test_index in kf.split(data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            strategy = TradingStrategy(
                target_percent=params['target_percent'],
                interval_step=params['interval_step'],
                rsi_oversold=params['rsi_oversold'],
                rsi_overbought=params['rsi_overbought'],
                leverage=params['leverage'],
                risk_per_trade=params['risk_per_trade'],
                bollinger_window=params['bollinger_window'],
                num_std_dev=params['num_std_dev'],
                scale_factor=params['scale_factor'],  # New parameter
                partial_take_profit=params['partial_take_profit']  # New parameter
            )

            strategy.execute(train_data)
            sharpe_ratio = strategy.calculate_sharpe_ratio()
            sharpe_ratios.append(sharpe_ratio)

    return np.mean(sharpe_ratios)  # Return the average Sharpe ratio across folds

def objective_function(params):
    param_dict = {
        'target_percent': params[0],
        'interval_step': 2,  # Fixed value
        'rsi_oversold': params[1],
        'rsi_overbought': params[2],
        'leverage': params[3],
        'risk_per_trade': params[4],
        'rsi_window': 14,
        'bollinger_window': 20,
        'num_std_dev': 2,
        'scale_factor': params[5],  # New parameter
        'partial_take_profit': params[6]  # New parameter
    }
    
    mean_sharpe_ratio = k_fold_cross_validation(file_paths, '2023-01-20', '2024-08-01', param_dict, k=5)
    return -mean_sharpe_ratio  # Negative for minimization

search_space = [
    Real(0.01, 0.1, name='target_percent'),
    Integer(10, 35, name='rsi_oversold'),
    Integer(65, 90, name='rsi_overbought'),
    Real(1, 20, name='leverage'),
    Real(0.1, 0.5, name='risk_per_trade'),
    Real(0.1, 1.0, name='scale_factor'),  # New search space for scale_factor
    Real(0.1, 1.0, name='partial_take_profit')  # New search space for partial_take_profit
]

res = gp_minimize(objective_function, search_space, n_calls=50, random_state=0)
print(f"Best parameters found: {res.x}")
print(f"Best objective value (Sharpe Ratio): {-res.fun}")

best_params = {
    'target_percent': res.x[0],
    'interval_step': 2,  # Fixed value
    'rsi_oversold': res.x[1],
    'rsi_overbought': res.x[2],
    'leverage': res.x[3],
    'risk_per_trade': res.x[4],
    'rsi_window': 14,
    'bollinger_window': 20,
    'num_std_dev': 2,
    'scale_factor': res.x[5],
    'partial_take_profit': res.x[6]
}

# Example data fetch and strategy execution
data = fetch_historical_data(file_paths[0], '2023-01-20', '2024-08-01')
data = calculate_rsi(data, window=best_params['rsi_window'])

strategy = TradingStrategy(
    target_percent=best_params['target_percent'],
    interval_step=best_params['interval_step'],
    rsi_oversold=best_params['rsi_oversold'],
    rsi_overbought=best_params['rsi_overbought'],
    leverage=best_params['leverage'],
    risk_per_trade=best_params['risk_per_trade'],
    bollinger_window=best_params['bollinger_window'],
    num_std_dev=best_params['num_std_dev'],
    scale_factor=best_params['scale_factor'],
    partial_take_profit=best_params['partial_take_profit']
)

strategy.execute(data)
final_balance = strategy.balance
print(f"Final balance with best parameters: {final_balance}")