import datetime
import pandas as pd
from binance.client import Client
import ta
import plotly.graph_objects as go

# Replace with your API key and secret
API_KEY = 
API_SECRET =

client = Client(API_KEY, API_SECRET)

def fetch_historical_data(symbol, interval, start_str, end_str):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    return klines

def calculate_macd_histogram(data):
    macd = ta.trend.MACD(data['close'])
    data['macd_hist'] = macd.macd_diff()
    return data

def main():
    symbol = "ADAUSDT"
    interval = Client.KLINE_INTERVAL_1MINUTE
    start_str = "1 Jul, 2024"
    end_str = "22 Jul, 2024"

    historical_prices = fetch_historical_data(symbol, interval, start_str, end_str)

    # Extract timestamps and OHLCV data
    timestamps = [datetime.datetime.fromtimestamp(price[0] / 1000) for price in historical_prices]
    closes = [float(price[4]) for price in historical_prices]

    # Create a DataFrame for MACD histogram calculation
    data = pd.DataFrame({
        'timestamp': timestamps,
        'close': closes
    })

    # Calculate MACD histogram
    data = calculate_macd_histogram(data)

    # Set pandas option to display all rows
    pd.set_option('display.max_rows', None)

    print(data[['timestamp', 'macd_hist']])

    # Plot MACD histogram and ADAUSDT price using Plotly
    fig = go.Figure()

    # Add MACD histogram
    fig.add_trace(go.Bar(
        x=data['timestamp'],
        y=data['macd_hist'],
        name='MACD Histogram',
        marker_color='blue',
        yaxis='y'
    ))

    # Add ADAUSDT price
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['close'],
        mode='lines',
        name='ADAUSDT Price',
        line=dict(color='orange'),
        yaxis='y2'
    ))

    # Update layout for dual y-axes
    fig.update_layout(
        title='MACD Histogram and ADAUSDT Price',
        xaxis_title='Date',
        yaxis=dict(
            title='MACD Histogram',
            side='left'
        ),
        yaxis2=dict(
            title='ADAUSDT Price',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )

    fig.show()

if __name__ == "__main__":
    main()