import pandas as pd
import talib
import itertools
import random
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
import os

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


class TradingStrategy:
    def __init__(self, target_percent: float, interval_step: int, rsi_oversold: float, rsi_overbought: float, leverage: float, risk_per_trade: float):
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

    def open_position(self, price: float, position_type: str, current_rsi: float, index: pd.Timestamp):
        self.position = (self.balance * self.leverage * self.risk_per_trade) / price
        self.entry_price = price
        self.position_type = position_type
        self.last_trade_index = index
        print(f"Opening {position_type} position at {price} on {index}, balance: {self.balance}, entry RSI: {current_rsi}")

    def close_position(self, price: float, profit: float, index: pd.Timestamp):
        self.balance += profit
        print(f"Closing {self.position_type} position at {price} on {index}, profit: {profit}, new balance: {self.balance}")
        self.position = 0

    def liquidate_position(self, price: float, loss: float, index: pd.Timestamp):
        self.balance -= loss
        print(f"Liquidating {self.position_type} position at {price} on {index}, loss: {loss}, new balance: {self.balance}")
        self.position = 0

    def execute(self, data: pd.DataFrame) -> pd.Series:
        prev_rsi = data['rsi'].iloc[0]

        for current_index, (index, row) in enumerate(data.iterrows()):
            current_rsi = row['rsi']

            if self.position == 0 and current_index >= self.last_trade_index + self.interval_step:
                if current_rsi <= self.rsi_oversold and prev_rsi > self.rsi_oversold:
                    self.open_position(row['close'], 'long', current_rsi, current_index)
                elif current_rsi >= self.rsi_overbought and prev_rsi < self.rsi_overbought:
                    self.open_position(row['close'], 'short', current_rsi, current_index)

            elif self.position != 0:
                if self.position_type == 'long':
                    if row['close'] >= self.entry_price * (1 + self.target_percent):
                        profit = self.position * (row['close'] - self.entry_price)
                        self.close_position(row['close'], profit, current_index)
                    elif row['close'] <= self.entry_price * (1 - (1 / self.leverage)):
                        loss = self.position * (self.entry_price - row['close'])
                        self.liquidate_position(row['close'], loss, current_index)

                elif self.position_type == 'short':
                    if row['close'] <= self.entry_price * (1 - self.target_percent):
                        profit = self.position * (self.entry_price - row['close'])
                        self.close_position(row['close'], profit, current_index)
                    elif row['close'] >= self.entry_price * (1 + (1 / self.leverage)):
                        loss = self.position * (row['close'] - self.entry_price)
                        self.liquidate_position(row['close'], loss, current_index)

            self.portfolio_values.append(self.balance)
            prev_rsi = current_rsi

        data['portfolio_value'] = self.portfolio_values
        return data['portfolio_value']

def grid_search(file_paths: List[str], start_date: str, end_date: str, param_grid: Dict[str, List[Any]], sample_length_days: int = 30, num_samples: int = 10) -> List[Dict[str, Any]]:
    results = []
    param_combinations = list(itertools.product(*param_grid.values()))

    for file_path in file_paths:
        for i, param_set in enumerate(param_combinations):
            params = dict(zip(param_grid.keys(), param_set))
            data = fetch_historical_data(file_path, start_date, end_date)
            data = calculate_rsi(data, window=params['rsi_window'])

            samples = generate_random_samples(data, sample_length_days, num_samples)
            sample_results = []

            for j, sample_data in enumerate(samples):
                strategy = TradingStrategy(
                    target_percent=params['target_percent'],
                    interval_step=params['interval_step'],
                    rsi_oversold=params['rsi_oversold'],
                    rsi_overbought=params['rsi_overbought'],
                    leverage=params['leverage'],
                    risk_per_trade=params['risk_per_trade']
                )
                portfolio_values = strategy.execute(sample_data)
                sample_results.append(strategy.balance)

                print(f"File: {file_path}, Permutation: {i + 1}/{len(param_combinations)}, Sample: {j + 1}/{num_samples}, Final Balance: {strategy.balance}")

            stats_result = perform_statistical_analysis(sample_results, data)
            results.append({
                'file_path': file_path,
                'params': params,
                'sample_results': sample_results,
                'statistical_summary': stats_result
            })

    return results

def generate_random_samples(data: pd.DataFrame, sample_length_days: int, num_samples: int) -> List[pd.DataFrame]:
    sample_length = pd.Timedelta(days=sample_length_days)
    max_start_date = data.index.max() - sample_length

    samples = []
    for _ in range(num_samples):
        random_start_date = data.index.min() + pd.Timedelta(days=random.randint(0, (max_start_date - data.index.min()).days))
        sample = data.loc[random_start_date:random_start_date + sample_length].copy()  # Ensure a copy is made to avoid SettingWithCopyWarning
        samples.append(sample)
    
    return samples

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

# Function to perform t-Test against a benchmark return
def t_test_against_benchmark(returns: List[float], benchmark_return: float = 0.0) -> Dict[str, float]:
    t_stat, p_value = stats.ttest_1samp(returns, benchmark_return)
    return {'t_stat': t_stat, 'p_value': p_value}

def perform_statistical_analysis(sample_results: List[float], data: pd.DataFrame) -> Dict[str, float]:
    buy_and_hold_return = calculate_buy_and_hold_return(data)
    mean_result = np.mean(sample_results)
    std_dev_result = np.std(sample_results)
    min_result = np.min(sample_results)
    max_result = np.max(sample_results)
    sharpe_ratio = calculate_sharpe_ratio(sample_results)
    t_test_results = t_test_against_benchmark(sample_results, buy_and_hold_return)

    return {
        'mean': mean_result,
        'std_dev': std_dev_result,
        'min': min_result,
        'max': max_result,
        'sharpe_ratio': sharpe_ratio,
        't_stat': t_test_results['t_stat'],
        'p_value': t_test_results['p_value'],
        'buy_and_hold_return': buy_and_hold_return
    }

def save_results_to_csv(results: List[Dict[str, Any]], file_name: str) -> None:
    flattened_results = []
    for result in results:
        flat_result = {
            'file_path': result['file_path'],
            **result['params'],
            'mean_return': result['statistical_summary']['mean'],
            'std_dev_return': result['statistical_summary']['std_dev'],
            'min_return': result['statistical_summary']['min'],
            'max_return': result['statistical_summary']['max'],
            'sharpe_ratio': result['statistical_summary']['sharpe_ratio'],
            't_stat': result['statistical_summary']['t_stat'],
            'p_value': result['statistical_summary']['p_value']
        }
        flattened_results.append(flat_result)

    df = pd.DataFrame(flattened_results)
    output_path = os.path.join(os.path.expanduser('~'), 'Downloads', file_name)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


file_paths = ['/Users/simonlavalle/Downloads/b-data/ADAUSDT_15min_data.csv']
param_grid = {
    'target_percent': [0.03, 0.04],
    'interval_step': [2],
    'rsi_oversold': [20, 25],
    'rsi_overbought': [75, 80],
    'leverage': [8, 10, 15],
    'risk_per_trade': [0.3, 0.5],
    'rsi_window': [14] 
}

results = grid_search(file_paths, '2024-04-20', '2024-08-04', param_grid)
save_results_to_csv(results, 'trading_strategy_results.csv')