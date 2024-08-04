import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import talib
import plotly.graph_objects as go
import datetime

API_KEY = 
API_SECRET = 
client = Client(API_KEY, API_SECRET)

INITIAL_BALANCE = 10000

def fetch_historical_data(symbol, interval, start_str, end_str):
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return data
    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"API Exception: {e}")
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
    last_trade_index = -interval_step  # Initialize to a negative index to allow the first trade

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

if __name__ == "__main__":
    symbol = 'ADAUSDT'
    start_str = '1 Jun 2023'
    end_str = '2 Sep 2023'

    intervals = [Client.KLINE_INTERVAL_1MINUTE, Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_1HOUR]
    interval_steps = {
        Client.KLINE_INTERVAL_1MINUTE: 1,
        Client.KLINE_INTERVAL_15MINUTE: 1,
        Client.KLINE_INTERVAL_4HOUR: 1  
    } 
    risk_per_trade_levels = [0.1, 0.25, 0.5]
    rsi_thresholds = [(20, 80), (25, 75), (30, 70)]
    target_percents = [0.04, 0.1, 0.15]
    leverage_levels = [5, 8, 10]

    for interval in intervals:
        data = fetch_historical_data(symbol, interval, start_str, end_str)
        if not data.empty:
            interval_step = interval_steps.get(interval, 1)  # Default to 1 if interval is not found
            print(f"Testing interval: {interval}, Step: {interval_step}")
            data_with_rsi = calculate_rsi(data.copy())
            results = grid_search(data_with_rsi, rsi_thresholds, target_percents, leverage_levels, risk_per_trade_levels, interval_step)
            best_params = max(results, key=results.get)
            print(f"Best parameters: {best_params}, with final balance: {results[best_params]}")
            plot_grid_search_results(results)
        else:
            print(f"Failed to fetch historical data for interval {interval}.")