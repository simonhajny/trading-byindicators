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

def calculate_rsi(data, window=14):
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=window).rsi()
    return data

def calculate_moving_averages(data, short_window=12, long_window=26):
    data['short_ma'] = data['close'].rolling(window=short_window).mean()
    data['long_ma'] = data['close'].rolling(window=long_window).mean()
    return data

def backtest_rsi_macd_ma_strategy(data, initial_balance=10000, rsi_buy_threshold=30, rsi_sell_threshold=70, short_ma_window=12, long_ma_window=26):
    balance = initial_balance
    position = 0
    portfolio_values = []
    buy_signals = []
    sell_signals = []

    data = calculate_rsi(data)
    data = calculate_moving_averages(data, short_window=short_ma_window, long_window=long_ma_window)

    for i in range(1, len(data)):
        if (data['rsi'].iloc[i] < rsi_buy_threshold and
            data['rsi'].iloc[i-1] >= rsi_buy_threshold and
            data['short_ma'].iloc[i] > data['long_ma'].iloc[i] and
            balance > 0):
            amount_to_invest = balance * 0.5
            position += amount_to_invest / data['close'].iloc[i]
            balance -= amount_to_invest
            buy_signals.append(i)
            print(f"Buy: {amount_to_invest / data['close'].iloc[i]} units at {data['close'].iloc[i]} on {data['timestamp'].iloc[i]}")

        elif (data['rsi'].iloc[i] > rsi_sell_threshold and
              data['rsi'].iloc[i-1] <= rsi_sell_threshold and
              data['short_ma'].iloc[i] < data['long_ma'].iloc[i] and
              position > 0):
            balance += position * data['close'].iloc[i]
            position = 0
            sell_signals.append(i)
            print(f"Sell: {balance / data['close'].iloc[i]} units at {data['close'].iloc[i]} on {data['timestamp'].iloc[i]}")

        portfolio_value = balance + position * data['close'].iloc[i]
        portfolio_values.append(portfolio_value)

    return portfolio_values, buy_signals, sell_signals

def calculate_baseline(data, initial_investment=10000):
    initial_price = data['close'].iloc[0]
    amount_held = initial_investment / initial_price
    baseline_values = amount_held * data['close']
    return baseline_values

def plot_portfolio_value(data, portfolio_values, baseline):
    fig = go.Figure()

    for interval, values in portfolio_values.items():
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=values,
            mode='lines',
            name=f'Portfolio Value ({interval})'
        ))

    fig.add_trace(go.Scatter(
        x=data['timestamp'],
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

def main(interval, rsi_params, ma_params):
    symbol = "ADAUSDT"
    start_str = "1 Feb, 2024"
    end_str = "28 Jul, 2024"

    historical_prices = fetch_historical_data(symbol, interval, start_str, end_str)

    timestamps = [datetime.datetime.fromtimestamp(price[0] / 1000) for price in historical_prices]
    opens = [float(price[1]) for price in historical_prices]
    highs = [float(price[2]) for price in historical_prices]
    lows = [float(price[3]) for price in historical_prices]
    closes = [float(price[4]) for price in historical_prices]

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })

    data = calculate_rsi(data, window=rsi_params['window'])
    data = calculate_moving_averages(data, short_window=ma_params['short_window'], long_window=ma_params['long_window'])

    # Fill missing values to avoid plotting issues
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    portfolio_value, buy_signals, sell_signals = backtest_rsi_macd_ma_strategy(
        data,
        rsi_buy_threshold=rsi_params['buy_threshold'],
        rsi_sell_threshold=rsi_params['sell_threshold'],
        short_ma_window=ma_params['short_window'],
        long_ma_window=ma_params['long_window']
    )

    baseline = calculate_baseline(data)
    return data, portfolio_value, buy_signals, sell_signals, baseline

if __name__ == "__main__":
    intervals = [
        Client.KLINE_INTERVAL_1MINUTE,
        Client.KLINE_INTERVAL_1HOUR,
        Client.KLINE_INTERVAL_30MINUTE
    ]

    rsi_params = {'window': 14, 'buy_threshold': 30, 'sell_threshold': 70}
    ma_params = {'short_window': 12, 'long_window': 26}

    portfolio_values = {}
    baseline = None

    for interval in intervals:
        data, portfolio_value, buy_signals, sell_signals, baseline = main(interval, rsi_params, ma_params)
        portfolio_values[interval] = portfolio_value

    plot_portfolio_value(data, portfolio_values, baseline)