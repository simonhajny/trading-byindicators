import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import ta
import datetime
import plotly.graph_objects as go

API_KEY =
API_SECRET = 
client = Client(API_KEY, API_SECRET)

INITIAL_BALANCE = 10000
LEVERAGE = 10
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RISK_PER_TRADE = 0.5

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
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=window).rsi()
    return data

def calculate_baseline(data):
    initial_price = data['close'].iloc[0]
    amount_held = INITIAL_BALANCE / initial_price
    return amount_held * data['close']

def plot_portfolio_value(data, portfolio_values, baseline):
    fig = go.Figure()
    for params, values in portfolio_values.items():
        interval, target_percent = params
        fig.add_trace(go.Scatter(
            x=data.index,
            y=values,
            mode='lines',
            name=f'Interval: {interval}, Target: {target_percent}'
        ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=baseline,
        mode='lines',
        name='Baseline (Buy & Hold)',
        line=dict(dash='dash')
    ))
    fig.update_layout(
        title='Portfolio Value Comparison',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Portfolio Value'),
        template='plotly_dark'
    )
    fig.show()

def trading_strategy(data, target_percent):
    balance = INITIAL_BALANCE
    position = None
    entry_price = None
    position_type = None
    entry_rsi = None
    portfolio_values = []
    prev_rsi = None
    for index, row in data.iterrows():
        current_rsi = row['rsi']

        if prev_rsi is not None:
            if position is None:
                amount_to_invest = balance * RISK_PER_TRADE

                if prev_rsi <= RSI_OVERSOLD and current_rsi > RSI_OVERSOLD:
                    position = (amount_to_invest * LEVERAGE) / row['close']
                    entry_price = row['close']
                    position_type = 'long'
                    entry_rsi = current_rsi
                    print(f"Entering long position at {entry_price} on {index}, RSI: {entry_rsi}")

                elif prev_rsi >= RSI_OVERBOUGHT and current_rsi < RSI_OVERBOUGHT:
                    position = (amount_to_invest * LEVERAGE) / row['close']
                    entry_price = row['close']
                    position_type = 'short'
                    entry_rsi = current_rsi
                    print(f"Entering short position at {entry_price} on {index}, RSI: {entry_rsi}")

            else:
                if position_type == 'long':
                    if row['close'] >= entry_price * (1 + target_percent):
                        profit = position * (row['close'] - entry_price)
                        balance += profit
                        print(f"Closing long position at {row['close']} on {index}, profit: {profit}, new balance: {balance}, entry RSI: {entry_rsi}")
                        position = None

                    elif row['close'] <= entry_price * (1 - (1 / LEVERAGE)):
                        loss = position * (entry_price - row['close'])
                        balance -= loss
                        print(f"Liquidating long position at {row['close']} on {index}, loss: {loss}, new balance: {balance}, entry RSI: {entry_rsi}")
                        position = None

                elif position_type == 'short':
                    if row['close'] <= entry_price * (1 - target_percent):
                        profit = position * (entry_price - row['close'])
                        balance += profit
                        print(f"Closing short position at {row['close']} on {index}, profit: {profit}, new balance: {balance}, entry RSI: {entry_rsi}")
                        position = None

                    elif row['close'] >= entry_price * (1 + (1 / LEVERAGE)):
                        loss = position * (row['close'] - entry_price)
                        balance -= loss
                        print(f"Liquidating short position at {row['close']} on {index}, loss: {loss}, new balance: {balance}, entry RSI: {entry_rsi}")
                        position = None

        portfolio_values.append(balance)
        prev_rsi = current_rsi

    data['portfolio_value'] = portfolio_values
    return balance, data['portfolio_value']

if __name__ == "__main__":
    symbol = 'ADAUSDT'
    intervals = [Client.KLINE_INTERVAL_1HOUR, Client.KLINE_INTERVAL_30MINUTE, Client.KLINE_INTERVAL_15MINUTE]
    target_percents = [0.02, 0.04, 0.03]
    start_str = '1 Jul 2024'
    end_str = '1 Aug 2024'

    portfolio_values = {}
    for interval in intervals:
        data = fetch_historical_data(symbol, interval, start_str, end_str)
        if not data.empty:
            for target_percent in target_percents:
                data_with_rsi = calculate_rsi(data.copy())
                final_balance, strategy_portfolio_values = trading_strategy(data_with_rsi, target_percent)
                portfolio_values[(interval, target_percent)] = strategy_portfolio_values
                print(f"Final balance for interval {interval} and target {target_percent}: {final_balance}")
        else:
            print(f"Failed to fetch historical data for interval {interval}.")

    baseline = calculate_baseline(data)
    plot_portfolio_value(data, portfolio_values, baseline)