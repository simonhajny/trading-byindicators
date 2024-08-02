from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
import datetime
import ta
import pandas as pd
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

API_KEY = '4XsWLW0VyKwXU7L0Skk3u53IdPhH5TkHWmVbtN6QD96FJXhzy4V1VxHeeXIQ7Nxt'
API_SECRET = 'lMA60aa2KDCYDM4JEEapTr2IcPk7VXkJelzkX9E6hfmPOUdPXsAq8o6NyGwJr84K'
client = Client(API_KEY, API_SECRET)

def fetch_historical_data(symbol, interval, start_str, end_str):
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        return klines
    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"API Exception: {e}")
        return []

def calculate_rsi(data, window=14):
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=window).rsi()
    return data

def should_open_position(rsi, oversold, overbought):
    if rsi < oversold:
        return "buy"
    elif rsi > overbought:
        return "sell"
    return None

def calculate_liquidation_price(entry_price, leverage, maintenance_margin_rate=0.005):
    liquidation_price = entry_price * (1 - maintenance_margin_rate / leverage)
    print(f"Calculated liquidation price: {liquidation_price}")
    return liquidation_price

def calculate_position_size(balance, price, leverage, risk_per_trade):
    position_size_ADA = (balance * risk_per_trade * leverage) / price
    return position_size_ADA

def backtest_rsi_strategy(data, initial_balance, oversold, overbought, leverage, risk_per_trade):
    balance = initial_balance
    position = 0
    entry_price = 0
    buy_signals = []
    sell_signals = []

    for i in range(1, len(data)):
        rsi = data['rsi'].iloc[i]
        signal = should_open_position(rsi, oversold, overbought)
        price = data['close'].iloc[i]
        timestamp = data['timestamp'].iloc[i]

        if signal == "buy" and balance > 0 and position == 0:
            position_size_ADA = calculate_position_size(balance, price, leverage, risk_per_trade)
            position += position_size_ADA
            balance -= balance * risk_per_trade
            entry_price = price
            buy_signals.append({
                'timestamp': timestamp,
                'quantity': position_size_ADA,
                'price': price,
                'dollars': position_size_ADA * price / leverage
            })

        elif position > 0 and price >= entry_price * 1.05:
            sell_amount = position
            balance += sell_amount * price / leverage
            position -= sell_amount
            sell_signals.append({
                'timestamp': timestamp,
                'quantity': sell_amount,
                'price': price,
                'dollars': sell_amount * price / leverage
            })

    final_portfolio_value = balance + position * data['close'].iloc[-1] / leverage if position > 0 else balance

    return final_portfolio_value, buy_signals, sell_signals



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













def main():
    historical_prices = fetch_historical_data(symbol = 'ADAUSDT', interval = Client.KLINE_INTERVAL_30MINUTE, start_str = '1 Feb 2023', end_str = '1 Aug 2024')
    if not historical_prices:
        print("Failed to fetch historical data.")
        return

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

    data = calculate_rsi(data)
    final_portfolio_value, buy_signals, sell_signals = backtest_rsi_strategy(data, initial_balance=10000, oversold=20, overbought=70, leverage=15, risk_per_trade=0.5)

    print("Final Portfolio Value:", final_portfolio_value)
    print("Buy Signals:")
    for signal in buy_signals:
        print(f"Time: {signal['timestamp']}, Quantity: {signal['quantity']}, Price: {signal['price']}, Dollars: {signal['dollars']}")
    print("Sell Signals:")
    for signal in sell_signals:
        print(f"Time: {signal['timestamp']}, Quantity: {signal['quantity']}, Price: {signal['price']}, Dollars: {signal['dollars']}")

if __name__ == "__main__":
    main()



def main(interval):
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


    portfolio_value, buy_signals, sell_signals = backtest_rsi_macd_ma_strategy(
        data,
        rsi_buy_threshold=rsi_params['buy_threshold'],
        rsi_sell_threshold=rsi_params['sell_threshold'],
    )

    baseline = calculate_baseline(data)
    return data, portfolio_value, buy_signals, sell_signals, baseline

if __name__ == "__main__":
    intervals = [
        Client.KLINE_INTERVAL_1MINUTE,
        Client.KLINE_INTERVAL_1HOUR,
        Client.KLINE_INTERVAL_30MINUTE
    ]

    portfolio_values = {}
    baseline = None

    for interval in intervals:
        data, portfolio_value, buy_signals, sell_signals, baseline = main(interval)
        portfolio_values[interval] = portfolio_value

    plot_portfolio_value(data, portfolio_values, baseline)


