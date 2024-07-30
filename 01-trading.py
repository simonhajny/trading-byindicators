# File: tradinimport binance.client
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import datetime
import pandas as pd

import ta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

API_KEY = 
API_SECRET = 
client = Client(API_KEY, API_SECRET)

def fetch_historical_data(symbol, interval, start_str, end_str):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    return klines

import pandas as pd
import ta

def calculate_rsi(data, window=14):
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=window).rsi()
    return data

def calculate_macd_histogram(data):
    macd = ta.trend.MACD(data['close'])
    data['macd_hist'] = macd.macd_diff()
    return data

def backtest_rsi_macd_strategy(data, initial_balance=10000, rsi_buy_threshold=30, rsi_sell_threshold=70, macd_histogram_buy_threshold=-0.0003, macd_histogram_sell_threshold=0.0003):
    balance = initial_balance
    position = 0
    portfolio_values = []
    buy_signals = []
    sell_signals = []

    data = calculate_rsi(data)
    data = calculate_macd_histogram(data)

    for i in range(1, len(data)):
        # Buy with 50% of balance
        if (data['rsi'].iloc[i] < rsi_buy_threshold and data['macd_hist'].iloc[i] <= macd_histogram_buy_threshold and balance > 0):
            amount_to_invest = balance * 0.5
            position += amount_to_invest / data['close'].iloc[i]
            balance -= amount_to_invest
            buy_signals.append(i)
            print(f"Buy: {amount_to_invest / data['close'].iloc[i]} units at {data['close'].iloc[i]} on {data['timestamp'].iloc[i]}")

        # Sell all position
        elif (data['rsi'].iloc[i] > rsi_sell_threshold and data['macd_hist'].iloc[i] >= macd_histogram_sell_threshold and position > 0):
            balance += position * data['close'].iloc[i]
            position = 0
            sell_signals.append(i)
            print(f"Sell: {balance} at {data['close'].iloc[i]} on {data['timestamp'].iloc[i]}")

        portfolio_values.append(balance + position * data['close'].iloc[i])

    final_value = balance + position * data['close'].iloc[-1]
    print(f"Final portfolio value: {final_value}")
    return portfolio_values, buy_signals, sell_signals






def plot_portfolio_value(data, portfolio_value, buy_signals, sell_signals):
    fig = go.Figure()

    # Add portfolio value trace with secondary y-axis
    fig.add_trace(go.Scatter(x=data['timestamp'], y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='orange'), yaxis='y2'))
    # Add price trace
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['close'], mode='lines', name='Price', line=dict(color='white')))

    # Add buy signals
    for i in buy_signals:
        fig.add_shape(
            type="line",
            x0=data['timestamp'].iloc[i],
            y0=data['close'].iloc[i] * 0.95,
            x1=data['timestamp'].iloc[i],
            y1=data['close'].iloc[i],
            line=dict(color="green", width=2),
            xref="x",
            yref="y"
        )
        fig.add_annotation(
            x=data['timestamp'].iloc[i],
            y=data['close'].iloc[i] * 0.95,
            text="Buy",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-10,
            font=dict(color="green")
        )

    # Add sell signals
    for i in sell_signals:
        fig.add_shape(
            type="line",
            x0=data['timestamp'].iloc[i],
            y0=data['close'].iloc[i] * 1.05,
            x1=data['timestamp'].iloc[i],
            y1=data['close'].iloc[i],
            line=dict(color="red", width=2),
            xref="x",
            yref="y"
        )
        fig.add_annotation(
            x=data['timestamp'].iloc[i],
            y=data['close'].iloc[i] * 1.05,
            text="Sell",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=10,
            font=dict(color="red")
        )

    # Update layout for secondary y-axis
    fig.update_layout(
        yaxis=dict(
            title='Price',
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange')
        ),
        yaxis2=dict(
            title='Portfolio Value',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            overlaying='y',
            side='right'
        ),
        xaxis=dict(title='Timestamp'),
        template='plotly_dark'
    )

    fig.show()

def main():
    symbol = "ADAUSDT"
    interval = Client.KLINE_INTERVAL_30MINUTE
    start_str = "1 Jan, 2022"
    end_str = "28 Jul, 2024"

    historical_prices = fetch_historical_data(symbol, interval, start_str, end_str)

    # Extract timestamps and OHLCV data
    timestamps = [datetime.datetime.fromtimestamp(price[0] / 1000) for price in historical_prices]
    opens = [float(price[1]) for price in historical_prices]
    highs = [float(price[2]) for price in historical_prices]
    lows = [float(price[3]) for price in historical_prices]
    closes = [float(price[4]) for price in historical_prices]

    # Create a DataFrame for RSI calculation
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })

    # Calculate RSI using the ta library
    data = calculate_rsi(data)
    data = calculate_macd_histogram(data)

    portfolio_value, buy_signals, sell_signals = backtest_rsi_macd_strategy(data)
    plot_portfolio_value(data, portfolio_value, buy_signals, sell_signals)

if __name__ == "__main__":
    main()
