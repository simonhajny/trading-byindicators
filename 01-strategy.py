import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import talib
import plotly.graph_objects as go
import datetime
import csv
import sys
import pandas as pd
from datetime import datetime, timedelta
import random

INITIAL_BALANCE = 10000
class DualWriter:
    def __init__(self, csv_writer):
        self.csv_writer = csv_writer
        self.console = sys.stdout
    def write(self, message):
        self.console.write(message)
        if message.strip():  # Avoid writing empty lines
            self.csv_writer.writerow([message.strip()])
    def flush(self):
        self.console.flush()

def fetch_historical_data(file_path, start_str, end_str):
    try:
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        mask = (data.index >= start_str) & (data.index <= end_str)
        filtered_data = data.loc[mask]
        
        return filtered_data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def calculate_rsi(data, window=14):
    data['rsi'] = talib.RSI(data['close'], timeperiod=window)
    return data

def trading_strategy(data, target_percent, interval_step, RSI_OVERSOLD, RSI_OVERBOUGHT, LEVERAGE, RISK_PER_TRADE):
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    portfolio_values = []
    position_type = None
    prev_rsi = data['rsi'].iloc[0]
    last_trade_index = -interval_step  

    for current_index, (index, row) in enumerate(data.iterrows()):
        current_rsi = row['rsi']

        if position == 0 and current_index >= last_trade_index + interval_step:  # Ensure position is zero and minimum interval has passed
            if current_rsi <= RSI_OVERSOLD and prev_rsi > RSI_OVERSOLD:
                position = (balance * LEVERAGE * RISK_PER_TRADE) / row['close']
                entry_price = row['close']
                position_type = 'long'
                last_trade_index = current_index
                print(f"Opening long position at {row['close']} on {index}, balance: {balance}, entry RSI: {current_rsi}")

            elif current_rsi >= RSI_OVERBOUGHT and prev_rsi < RSI_OVERBOUGHT:
                position = (balance * LEVERAGE * RISK_PER_TRADE) / row['close']
                entry_price = row['close']
                position_type = 'short'
                last_trade_index = current_index
                print(f"Opening short position at {row['close']} on {index}, balance: {balance}, entry RSI: {current_rsi}")

        elif position != 0:
            if position_type == 'long':
                if row['close'] >= entry_price * (1 + target_percent):
                    profit = position * (row['close'] - entry_price)
                    balance += profit
                    print(f"Closing long position at {row['close']} on {index}, profit: {profit}, new balance: {balance}, entry RSI: {current_rsi}")
                    position = 0  # Set position to zero after closing

                elif row['close'] <= entry_price * (1 - (1 / (LEVERAGE))):
                    loss = position * (entry_price - row['close'])
                    balance -= loss
                    print(f"Liquidating long position at {row['close']} on {index}, loss: {loss}, new balance: {balance}, entry RSI: {current_rsi}")
                    position = 0  # Set position to zero after liquidating

            elif position_type == 'short':
                if row['close'] <= entry_price * (1 - target_percent):
                    profit = position * (entry_price - row['close'])
                    balance += profit
                    print(f"Closing short position at {row['close']} on {index}, profit: {profit}, new balance: {balance}, entry RSI: {current_rsi}")
                    position = 0  # Set position to zero after closing

                elif row['close'] >= entry_price * (1 + (1 / (LEVERAGE))):
                    loss = position * (row['close'] - entry_price)
                    balance -= loss
                    print(f"Liquidating short position at {row['close']} on {index}, loss: {loss}, new balance: {balance}, entry RSI: {current_rsi}")
                    position = 0  # Set position to zero after liquidating

        portfolio_values.append(balance)
        prev_rsi = current_rsi

    data['portfolio_value'] = portfolio_values
    return balance, data['portfolio_value']

def grid_search(data, rsi_thresholds, target_percents, leverage_levels, risk_per_trade_levels, interval_step):
    results = {}
    for rsi_oversold, rsi_overbought in rsi_thresholds:
        for target_percent in target_percents:
            for leverage in leverage_levels:
                for risk_per_trade in risk_per_trade_levels:
                    final_balance, portfolio_values = trading_strategy(data, target_percent, interval_step, rsi_oversold, rsi_overbought, leverage, risk_per_trade)
                    key = (rsi_oversold, rsi_overbought, target_percent, leverage, risk_per_trade)
                    results[key] = final_balance
                    print(f"RSI: ({rsi_oversold}, {rsi_overbought}), Target: {target_percent}, Leverage: {leverage}, Risk: {risk_per_trade}, Final Balance: {final_balance}")
    return results

def plot_grid_search_results(results):
    fig = go.Figure()
    for params, balance in results.items():
        rsi_oversold, rsi_overbought, target_percent, leverage, risk_per_trade = params
        fig.add_trace(go.Bar(
            x=[f"RSI: ({rsi_oversold}, {rsi_overbought}), Target: {target_percent}, Leverage: {leverage}, Risk: {risk_per_trade}"],
            y=[balance],
            name=f"RSI: ({rsi_oversold}, {rsi_overbought}), Target: {target_percent}, Leverage: {leverage}, Risk: {risk_per_trade}"
        ))
    fig.update_layout(title='Grid Search Results', xaxis_title='Parameters', yaxis_title='Final Balance')
    fig.show()



def generate_random_date_ranges(start_str, end_str, n, interval_length=30):
    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_str, '%Y-%m-%d')
    max_start_date = end_date - timedelta(days=interval_length)
    date_ranges = []
    for _ in range(n):
        random_start = start_date + timedelta(days=random.randint(0, (max_start_date - start_date).days))
        random_end = random_start + timedelta(days=interval_length)
        date_ranges.append((random_start.strftime('%Y-%m-%d'), random_end.strftime('%Y-%m-%d')))
    return date_ranges

def aggregate_results(all_results):
    aggregated = {}
    for result in all_results:
        for params, balance in result.items():
            if params not in aggregated:
                aggregated[params] = []
            aggregated[params].append(balance)
    return {params: sum(balances) / len(balances) for params, balances in aggregated.items()}

if __name__ == "__main__":
    file_paths = {
        '1min': '/Users/simonlavalle/Downloads/b-data/ADAUSDT_1min_data.csv',
        '15min': '/Users/simonlavalle/Downloads/b-data/ADAUSDT_15min_data.csv',
        '1h': '/Users/simonlavalle/Downloads/b-data/ADAUSDT_1h_data.csv'
    }
    start_str = '2021-01-01'
    end_str = '2024-08-04'
    n_samples = 5  # Number of samples

    intervals = ['1min', '15min', '1h']
    interval_steps = {
        '1min': 2,
        '15min': 2,
        '1h': 2
    }
    risk_per_trade_levels = [0.25, 0.5, 0.7]
    rsi_thresholds = [(20, 80), (25, 75), (30, 70)]
    target_percents = [0.1, 0.15, 0.2]
    leverage_levels = [8, 10, 15]
    output_csv_path = '/Users/simonlavalle/Downloads/output.csv'

    date_ranges = generate_random_date_ranges(start_str, end_str, n_samples)

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        sys.stdout = DualWriter(writer)

        for interval in intervals:
            file_path = file_paths[interval]
            all_results = []
            for start, end in date_ranges:
                data = fetch_historical_data(file_path, start, end)
                if not data.empty:
                    interval_step = interval_steps.get(interval, 1)  # Default to 1 if interval is not found
                    print(f"Testing interval: {interval}, Step: {interval_step}, Date Range: {start} to {end}")
                    data_with_rsi = calculate_rsi(data.copy())
                    results = grid_search(data_with_rsi, rsi_thresholds, target_percents, leverage_levels, risk_per_trade_levels, interval_step)
                    all_results.append(results)
                else:
                    print(f"Failed to fetch historical data for interval {interval}, Date Range: {start} to {end}")

            if all_results:
                aggregated_results = aggregate_results(all_results)
                best_params = max(aggregated_results, key=aggregated_results.get)
                print(f"Best parameters: {best_params}, with average final balance: {aggregated_results[best_params]}")
                plot_grid_search_results(aggregated_results)
                
                for params, final_balance in aggregated_results.items():
                    rsi_oversold, rsi_overbought, target_percent, leverage, risk_per_trade = params
                    print(f"RSI Oversold: {rsi_oversold}, RSI Overbought: {rsi_overbought}, Target: {target_percent}, Leverage: {leverage}, Risk: {risk_per_trade}, Final Balance: {final_balance}")