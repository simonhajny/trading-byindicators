import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import logging
from typing import List, Tuple
from scipy.stats import entropy
from scipy.signal import find_peaks

def fetch_historical_klines(symbol: str, interval: str, start_str: str, end_str: str = None, limit: int = 1000) -> pd.DataFrame:
    try:
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_ts}&limit={limit}"
        if end_ts:
            url += f"&endTime={end_ts}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        klines = response.json()
        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        data = pd.DataFrame(klines, columns=columns)
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                        'Taker buy base asset volume', 'Taker buy quote asset volume']
        data[numeric_cols] = data[numeric_cols].astype(float)
        return data
    except requests.RequestException as e:
        logging.error(f"Error fetching historical klines: {e}")
        return pd.DataFrame()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def combine_rsi(data: pd.Series, windows: List[int]) -> pd.DataFrame:
    rsi_data = pd.DataFrame()
    for window in windows:
        rsi_data[f'RSI_{window}'] = calculate_rsi(data, window)

    rsi_data['RSI_14'] = calculate_rsi(data, 14)
    rsi_data['RSI_Simple_Avg'] = rsi_data[[f'RSI_{w}' for w in windows]].mean(axis=1)
    weights = np.array(range(len(windows), 0, -1))
    rsi_data['RSI_Weighted_Avg'] = np.average(rsi_data[[f'RSI_{w}' for w in windows]], axis=1, weights=weights)
    exp_weights = np.exp(range(len(windows)))
    rsi_data['RSI_Exp_Weighted_Avg'] = np.average(rsi_data[[f'RSI_{w}' for w in windows]], axis=1, weights=exp_weights)

    return rsi_data

def analyze_rsi_crossings(rsi_data: pd.DataFrame, threshold: float = 25) -> pd.DataFrame:
    crossings = pd.DataFrame(index=rsi_data.index)
    for column in rsi_data.columns:
        crossings[f'{column}_Lower'] = (rsi_data[column] < threshold) & (rsi_data[column].shift(1) >= threshold)
        crossings[f'{column}_Upper'] = (rsi_data[column] > (100 - threshold)) & (rsi_data[column].shift(1) <= (100 - threshold))
    return crossings

def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def calculate_chaikin_volatility(data: pd.DataFrame, high_col: str = 'High', low_col: str = 'Low', ema_window: int = 10, roc_window: int = 10) -> pd.Series:
    high_low_diff = data[high_col] - data[low_col]
    ema_diff = calculate_ema(high_low_diff, ema_window)
    return ((ema_diff.diff(roc_window)) / ema_diff.shift(roc_window)) * 100

def analyze_chaikin_crossings(chaikin_volatility: pd.Series, threshold: float = 10) -> pd.Series:
    return (chaikin_volatility > threshold) & (chaikin_volatility.shift(1) <= threshold)

def calculate_signal_entropy(signal: pd.Series, bins: int = 10) -> float:
    signal = signal.dropna()  # Ensure no NaNs are present
    hist, _ = np.histogram(signal, bins=bins)
    prob_dist = hist / hist.sum()
    return entropy(prob_dist)

def find_signal_peaks(signal: pd.Series, prominence: float = 1.0) -> np.ndarray:
    return find_peaks(signal, prominence=prominence)[0]

def calculate_signal_to_noise_ratio(signal: pd.Series) -> float:
    return np.mean(signal) / np.std(signal)

def analyze_information_theory(rsi_data: pd.DataFrame, chaikin_volatility: pd.Series) -> pd.DataFrame:
    info_analysis = pd.DataFrame(index=['RSI_14'] + list(rsi_data.columns[-3:]) + ['CVI'],
                                 columns=['Entropy', 'Peak_Count', 'SNR'])
    
    for column in info_analysis.index:
        signal = chaikin_volatility if column == 'CVI' else rsi_data[column]
        info_analysis.loc[column, 'Entropy'] = calculate_signal_entropy(signal)
        info_analysis.loc[column, 'Peak_Count'] = len(find_signal_peaks(signal))
        info_analysis.loc[column, 'SNR'] = calculate_signal_to_noise_ratio(signal)
    
    return info_analysis

def analyze_signal_quality(price: pd.Series, rsi_data: pd.DataFrame, crossings: pd.DataFrame, chaikin_volatility: pd.Series, chaikin_crossings: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_quality = pd.DataFrame(index=['RSI_14', 'RSI_Simple_Avg', 'RSI_Weighted_Avg', 'RSI_Exp_Weighted_Avg', 'CVI'], 
                                  columns=['Total_Signals', 'Correct_Signals', 'Accuracy'])

    future_window = 10  # Number of periods to look ahead for price movement

    for column in ['RSI_14', 'RSI_Simple_Avg', 'RSI_Weighted_Avg', 'RSI_Exp_Weighted_Avg']:
        lower_signals = crossings[f'{column}_Lower']
        upper_signals = crossings[f'{column}_Upper']

        total_signals = lower_signals.sum() + upper_signals.sum()
        correct_lower = ((price.shift(-future_window) > price) & lower_signals).sum()
        correct_upper = ((price.shift(-future_window) < price) & upper_signals).sum()

        signal_quality.loc[column, 'Total_Signals'] = total_signals
        signal_quality.loc[column, 'Correct_Signals'] = correct_lower + correct_upper
        signal_quality.loc[column, 'Accuracy'] = (correct_lower + correct_upper) / total_signals if total_signals > 0 else 0

    cvi_total_signals = chaikin_crossings.sum()
    cvi_correct_signals = ((price.shift(-future_window) > price) & chaikin_crossings).sum()

    signal_quality.loc['CVI', 'Total_Signals'] = cvi_total_signals
    signal_quality.loc['CVI', 'Correct_Signals'] = cvi_correct_signals
    signal_quality.loc['CVI', 'Accuracy'] = cvi_correct_signals / cvi_total_signals if cvi_total_signals > 0 else 0

    correlations = pd.DataFrame(index=['Price', 'CVI'], columns=['RSI_14', 'RSI_Simple_Avg', 'RSI_Weighted_Avg', 'RSI_Exp_Weighted_Avg'])
    for column in correlations.columns:
        correlations.loc['Price', column] = price.corr(rsi_data[column])
        correlations.loc['CVI', column] = chaikin_volatility.corr(rsi_data[column])

    crossing_counts = pd.DataFrame(index=['RSI_14', 'RSI_Simple_Avg', 'RSI_Weighted_Avg', 'RSI_Exp_Weighted_Avg'],
                                   columns=['Lower_Crossings', 'Upper_Crossings'])

    for column in crossing_counts.index:
        crossing_counts.loc[column, 'Lower_Crossings'] = crossings[f'{column}_Lower'].sum()
        crossing_counts.loc[column, 'Upper_Crossings'] = crossings[f'{column}_Upper'].sum()

    return signal_quality, correlations, crossing_counts

def plot_trading_signals(price: pd.Series, rsi_data: pd.DataFrame, crossings: pd.DataFrame, chaikin_volatility: pd.Series, chaikin_crossings: pd.Series, cvi_threshold: float = 50.0):
    color_scheme = {
        'RSI_14': '#1f77b4',  # Blue
        'RSI_Simple_Avg': '#ff7f0e',  # Orange
        'RSI_Weighted_Avg': '#2ca02c',  # Green
        'RSI_Exp_Weighted_Avg': '#d62728',  # Red
        'Price': '#7f7f7f',  # Gray
        'CVI': '#9467bd',  # Purple
    }

    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=('Price and Signals', 'RSI Combinations', 'Chaikin Volatility Index'),
                        row_heights=[0.5, 0.3, 0.2])

    # Plot 1: Price and Signals
    fig.add_trace(go.Scatter(x=price.index, y=price, mode='lines', name='Price', 
                             line=dict(color=color_scheme['Price'], width=1)), row=1, col=1)

    # Find intervals where CVI is above the threshold
    above_threshold = chaikin_volatility > cvi_threshold
    intervals = above_threshold.astype(int).diff().fillna(0)  # Identify start and end of regions above threshold

    # Start and end indices of regions where CVI is above the threshold
    starts = price.index[intervals == 1]
    ends = price.index[intervals == -1]

    # If the last interval doesn't close, mark it until the end of the data
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([price.index[-1]]))

    # Shading the price chart between these intervals
    for start, end in zip(starts, ends):
        fig.add_vrect(x0=start, x1=end, fillcolor='rgba(148, 103, 189, 0.2)', 
                      layer='below', line_width=0)

    for column in ['RSI_14', 'RSI_Simple_Avg', 'RSI_Weighted_Avg', 'RSI_Exp_Weighted_Avg']:
        lower_crossings = price.index[crossings[f'{column}_Lower']]
        upper_crossings = price.index[crossings[f'{column}_Upper']]

        if not lower_crossings.empty:
            fig.add_trace(go.Scatter(x=lower_crossings, y=price.loc[lower_crossings],
                                     mode='markers', marker=dict(color=color_scheme[column], symbol='triangle-down', size=8),
                                     name=f'{column} Lower Crossing'), row=1, col=1)

        if not upper_crossings.empty:
            fig.add_trace(go.Scatter(x=upper_crossings, y=price.loc[upper_crossings],
                                     mode='markers', marker=dict(color=color_scheme[column], symbol='triangle-up', size=8),
                                     name=f'{column} Upper Crossing'), row=1, col=1)

    if chaikin_crossings.any():
        crossing_dates = chaikin_volatility.index[chaikin_crossings]
        fig.add_trace(go.Scatter(x=crossing_dates, y=price.loc[crossing_dates],
                                 mode='markers', marker=dict(color=color_scheme['CVI'], symbol='star', size=10),
                                 name='CVI Crossing'), row=1, col=1)

    # Plot 2: RSI Combinations
    for column in ['RSI_14', 'RSI_Simple_Avg', 'RSI_Weighted_Avg', 'RSI_Exp_Weighted_Avg']:
        fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data[column], mode='lines', name=column, 
                                 line=dict(color=color_scheme[column], width=1)), row=2, col=1)

    # Add RSI threshold lines
    fig.add_hline(y=25, line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'), row=2, col=1)
    fig.add_hline(y=75, line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'), row=2, col=1)

    # Plot 3: Chaikin Volatility Index
    fig.add_trace(go.Scatter(x=chaikin_volatility.index, y=chaikin_volatility, mode='lines', name='CVI', 
                             line=dict(color=color_scheme['CVI'], width=1)), row=3, col=1)

    # Add CVI threshold line
    fig.add_hline(y=cvi_threshold, line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'), row=3, col=1)

    fig.update_layout(
        title='',
        template="plotly_dark",
        height=1420,
        width=2525,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)')
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="CVI", row=3, col=1)

    fig.show()

def main():
    symbol = 'ADAUSDT'
    interval = '15m'
    start_date = '2024-03-19'
    end_date = '2024-04-20'
    
    data = fetch_historical_klines(symbol, interval, start_date, end_date)
    
    windows = [7, 14, 21, 28]  # RSI windows
    rsi_data = combine_rsi(data['Close'], windows)
    crossings = analyze_rsi_crossings(rsi_data, threshold=25)
    chaikin_volatility = calculate_chaikin_volatility(data)
    chaikin_crossings = analyze_chaikin_crossings(chaikin_volatility, threshold=10)
    plot_trading_signals(data['Close'], rsi_data, crossings, chaikin_volatility, chaikin_crossings)
    signal_quality, correlations, crossing_counts = analyze_signal_quality(data['Close'], rsi_data, crossings, chaikin_volatility, chaikin_crossings)

    print("Signal Quality Analysis:")
    print(signal_quality)
    print("\nCorrelations:")
    print(correlations)
    print("\nCrossing Counts:")
    print(crossing_counts)

    info_analysis = analyze_information_theory(rsi_data, chaikin_volatility)
    print("\nInformation Theory and Signal Processing Analysis:")
    print(info_analysis)

    best_indicator = signal_quality['Accuracy'].idxmax()
    most_frequent_signal = crossing_counts.sum(axis=1).idxmax()
    most_entropic_signal = info_analysis['Entropy'].idxmax()
    highest_snr_signal = info_analysis['SNR'].idxmax()

    print(f"\nBest performing indicator (by accuracy): {best_indicator}")
    print(f"Most frequent signal generator: {most_frequent_signal}")
    print(f"Most entropic (unpredictable) signal: {most_entropic_signal}")
    print(f"Highest signal-to-noise ratio: {highest_snr_signal}")


if __name__ == "__main__":
    main()