import numpy as np
import requests
import logging
from scipy.optimize import differential_evolution
from typing import Union, List, Tuple
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

ALPHA = 0.5       # Controls amplification of positive CVI
BETA = 1          # Controls dampening of negative CVI
LONG_THRESHOLD = -0.5
SHORT_THRESHOLD = 0.5

class SignalOptimizer:
    def __init__(self, processor: 'TradingSignalProcessor', signals_to_optimize: List[str]):
        self.processor = processor
        self.signals_to_optimize = signals_to_optimize

    def optimize_parameters(self) -> dict:
        optimized_params = {}
        for signal in self.signals_to_optimize:
            bounds = [(0, 10), (0, 10), (-0.8, -0.5), (0.5, 0.8)]  # ALPHA, BETA, LONG_THRESHOLD, SHORT_THRESHOLD
            result = differential_evolution(self.objective_function, bounds, args=(signal,), popsize=10, maxiter=1)
            optimized_params[signal] = {
                'ALPHA': result.x[0],
                'BETA': result.x[1],
                'LONG_THRESHOLD': result.x[2],
                'SHORT_THRESHOLD': result.x[3]
            }
        return optimized_params

    def objective_function(self, params: Tuple[float, float, float, float], signal: str) -> float:
        alpha, beta, long_threshold, short_threshold = params
        composite_signal = self.processor.composite_signal(self.processor.rsi_data[signal], self.processor.cvi, alpha, beta)
        trade_signals = self.processor.generate_trade_signals(composite_signal, short_threshold, long_threshold)
        signal_quality = self.processor.analyze_signal_quality(trade_signals)
        return -signal_quality.loc[signal, 'Accuracy']


class TradingSignalProcessor:
    def __init__(self, start_date: str, end_date: str, rsi_windows=[10, 14, 21]):
        self.start_date = start_date
        self.end_date = end_date
        self.rsi_windows = rsi_windows
        self.data = None
        self.rsi_data = None
        self.signals = None

    def _fetch_historical_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        all_klines = []
        start_ts = int(pd.to_datetime(self.start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(self.end_date).timestamp() * 1000)
        current_start = start_ts

        while current_start < end_ts:
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={current_start}&endTime={end_ts}&limit=1000"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_start = klines[-1][0] + 1  # Start from the next timestamp
                
            except requests.RequestException as e:
                logging.error(f"Error fetching historical klines for {symbol}: {e}")
                break

        if not all_klines:
            return pd.DataFrame()

        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        data = pd.DataFrame(all_klines, columns=columns)
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                        'Taker buy base asset volume', 'Taker buy quote asset volume']
        data[numeric_cols] = data[numeric_cols].astype(float)
        data.set_index('Open time', inplace=True)
        self.data = data
        return data

    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _combine_rsi(self) -> pd.DataFrame:
        rsi_data = pd.DataFrame(index=self.data.index)
        for window in self.rsi_windows:
            rsi_data[f'RSI_{window}'] = self._calculate_rsi(self.data['Close'], window)

        for column in rsi_data.columns:
            rsi_data[column] = self.normalize_rsi(rsi_data[column])
        rsi_data.columns = [f'RSI{w}' for w in self.rsi_windows]
        rsi_data['RSI_S'] = rsi_data.mean(axis=1)
        weights = np.array(range(len(self.rsi_windows), 0, -1))
        rsi_data['RSI_W'] = np.average(rsi_data[[f'RSI{w}' for w in self.rsi_windows]], axis=1, weights=weights)
        exp_weights = np.exp(range(len(self.rsi_windows)))
        rsi_data['RSI_E'] = np.average(rsi_data[[f'RSI{w}' for w in self.rsi_windows]], axis=1, weights=exp_weights)
        
        return rsi_data
    
    def normalize_rsi(self, rsi: pd.Series) -> pd.Series:
        return (rsi - 50) / 50 

    def calculate_cvi(self, ema_window: int = 10, roc_window: int = 10) -> pd.Series:
        high_low_diff = self.data['High'] - self.data['Low']
        ema_diff = high_low_diff.ewm(span=ema_window, adjust=False).mean()
        cvi = ((ema_diff - ema_diff.shift(roc_window)) / ema_diff.shift(roc_window)) * 100
        return cvi

    def composite_signal(self, rsi_norm: pd.Series, cvi: pd.Series, alpha: float, beta: float) -> pd.Series:
        # Normalize positive CVI to [0, 1] range with max at 100
        cvi_pos_norm = np.minimum(cvi / 100, 1)
        # Normalize negative CVI to [-1, 0] range
        cvi_neg_norm = np.maximum(cvi / 100, -1)
        # Create a modulation factor based on CVI
        cvi_factor = np.where(
            cvi >= 0,
            1 + alpha * np.power(cvi_pos_norm, 2),  # Amplify positive CVI, max at 100
            1 / (1 + beta * np.abs(cvi_neg_norm))   # Dampen negative CVI
        )
        # Modulate RSI based on CVI
        modulated_rsi = rsi_norm * cvi_factor
        return np.clip(modulated_rsi, -1, 1)

    def generate_trade_signals(self, signal, short_threshold: float, long_threshold: float) -> pd.DataFrame:
        if isinstance(signal, pd.Series):
            signal = pd.DataFrame(signal)
        
        signals = pd.DataFrame(index=signal.index, columns=signal.columns, data=0)
        last_signal = pd.Series(index=signal.columns, data=0)
        
        for i in range(len(signal)):
            for column in signal.columns:
                if signal.iloc[i][column] > short_threshold and last_signal[column] != -1:
                    signals.loc[signal.index[i], column] = -1
                    last_signal[column] = -1
                elif signal.iloc[i][column] < long_threshold and last_signal[column] != 1:
                    signals.loc[signal.index[i], column] = 1
                    last_signal[column] = 1
                elif long_threshold <= signal.iloc[i][column] <= short_threshold:
                    last_signal[column] = 0  # Reset last_signal in neutral range
                else:
                    signals.loc[signal.index[i], column] = 0   # No signal
        return signals

    def generate_rsi_trade_signals(self, rsi_norm: pd.Series) -> pd.Series:
        return self.generate_trade_signals(rsi_norm, 0.5, -0.5)

    def analyze_signal_quality(self, signals: pd.DataFrame, target_pct: float = 0.005, liquidation_pct: float = 0.03) -> pd.DataFrame:
        signal_quality = pd.DataFrame(index=signals.columns,
                                    columns=['Total_Signals', 'Accuracy', 'Signal_to_Noise'])
        price = self.data['Close']
        
        for column in signals.columns:
            long_signals = signals[column] == 1
            short_signals = signals[column] == -1
            total_signals = long_signals.sum() + short_signals.sum()
            
            correct_long = 0
            correct_short = 0

            # Iterate over each long trade signal
            for signal_time in price[long_signals].index:
                start_price = price.loc[signal_time]
                future_prices = price.loc[signal_time:].pct_change()
                max_favorable_movement = future_prices[future_prices > 0].max()
                max_unfavorable_movement = future_prices[future_prices < 0].min()

                if max_favorable_movement >= target_pct and abs(max_unfavorable_movement) <= liquidation_pct:
                    correct_long += 1

            # Iterate over each short trade signal
            for signal_time in price[short_signals].index:
                start_price = price.loc[signal_time]
                future_prices = price.loc[signal_time:].pct_change()
                max_favorable_movement = future_prices[future_prices < 0].min()  # Negative for short
                max_unfavorable_movement = future_prices[future_prices > 0].max()  # Positive for unfavorable

                if abs(max_favorable_movement) >= target_pct and max_unfavorable_movement <= liquidation_pct:
                    correct_short += 1

            signal_quality.loc[column, 'Total_Signals'] = total_signals
            signal_quality.loc[column, 'Accuracy'] = (correct_long + correct_short) / total_signals if total_signals > 0 else 0

            signal = signals[column].diff().abs().mean()
            noise = signals[column].diff().std()
            signal_quality.loc[column, 'Signal_to_Noise'] = signal / noise if noise != 0 else 0

        return signal_quality

    def get_trade_counts(self, signals: pd.DataFrame) -> pd.DataFrame:
        trade_counts = pd.DataFrame(index=signals.columns, columns=['Long_Trades', 'Short_Trades'])
        for column in signals.columns:
            trade_counts.loc[column, 'Long_Trades'] = (signals[column] == 1).sum()
            trade_counts.loc[column, 'Short_Trades'] = (signals[column] == -1).sum()
        return trade_counts
    
    def plot_signals(self, rsi_data: pd.DataFrame, cvi: pd.Series, composite_signals: pd.DataFrame, trade_signals: Union[pd.Series, pd.DataFrame]):
        global LONG_THRESHOLD, SHORT_THRESHOLD

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=["Price", "Normalized RSIs", "CVI", "Composite Signals", "Trade Signals"])

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06EC4']
        fig.add_trace(go.Candlestick(x=self.data.index, open=self.data['Open'], high=self.data['High'], 
                                    low=self.data['Low'], close=self.data['Close'], name='Price'), row=1, col=1)

        # Add trade markers to the price chart
        if isinstance(trade_signals, pd.Series):
            trade_signals = pd.DataFrame(trade_signals)

        for i, column in enumerate(trade_signals.columns):
            long_signals = trade_signals.index[trade_signals[column] == 1]
            short_signals = trade_signals.index[trade_signals[column] == -1]

            fig.add_trace(go.Scatter(
                x=long_signals, y=self.data.loc[long_signals, 'Low'],
                mode='markers', name=f'Long Signal ({column})',
                marker=dict(color=colors[i % len(colors)], size=10, symbol='triangle-up'),
                showlegend=False), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=short_signals, y=self.data.loc[short_signals, 'High'],
                mode='markers', name=f'Short Signal ({column})',
                marker=dict(color=colors[i % len(colors)], size=10, symbol='triangle-down'),
                showlegend=False), row=1, col=1)

        # Add RSI lines
        for i, column in enumerate(rsi_data.columns):
            fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data[column], mode='lines', 
                                    name=column, line=dict(color=colors[i % len(colors)], width=1)), row=2, col=1)
        
        fig.add_hline(y=-0.5, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
        fig.add_hline(y=0.5, line=dict(color='red', width=1, dash='dash'), row=2, col=1)

        # Add CVI
        fig.add_trace(go.Scatter(x=cvi.index, y=cvi, mode='lines', name='CVI', 
                                line=dict(color='white', width=1)), row=3, col=1)

        # Add composite signals
        for i, column in enumerate(composite_signals.columns):
            fig.add_trace(go.Scatter(x=composite_signals.index, y=composite_signals[column], mode='lines', 
                                    name=f'Comp_{column}', line=dict(color=colors[i % len(colors)], width=1)), row=4, col=1)
        
        fig.add_hline(y=LONG_THRESHOLD, line=dict(color='red', width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=SHORT_THRESHOLD, line=dict(color='red', width=1, dash='dash'), row=4, col=1)

        # Add trade signals subplot
        for i, column in enumerate(trade_signals.columns):
            long_signals = trade_signals.index[trade_signals[column] == 1]
            short_signals = trade_signals.index[trade_signals[column] == -1]

            fig.add_trace(go.Scatter(
                x=long_signals, y=[i+1]*len(long_signals),
                mode='markers', name=f'Long Signal ({column})',
                marker=dict(color=colors[i % len(colors)], size=8, symbol='triangle-up')), row=5, col=1)

            fig.add_trace(go.Scatter(
                x=short_signals, y=[i+1]*len(short_signals),
                mode='markers', name=f'Short Signal ({column})',
                marker=dict(color=colors[i % len(colors)], size=8, symbol='triangle-down')), row=5, col=1)

        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='black',
            paper_bgcolor='black',
            height=1500,
            title_font=dict(size=24, color='white'),
            legend=dict(font=dict(color='white')),
            xaxis_rangeslider_visible=False
        )

        for i in range(1, 6):
            fig.update_yaxes(title_font=dict(color='white'), tickfont=dict(color='white'), row=i, col=1)
            fig.update_xaxes(tickfont=dict(color='white'), row=i, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Normalized RSI", row=2, col=1)
        fig.update_yaxes(title_text="CVI", row=3, col=1)
        fig.update_yaxes(title_text="Composite Signal", row=4, col=1)
        fig.update_yaxes(title_text="Trade Signals", row=5, col=1)

        fig.show()

    def log_trade_signals(self, rsi_trade_signals: pd.DataFrame, composite_trade_signals: pd.DataFrame, filename: str = "trade_signals.csv"):
        logged_signals = []

        for column in rsi_trade_signals.columns:
            for timestamp, signal in rsi_trade_signals[column].items():
                if signal != 0:  # Only log non-zero signals
                    price = self.data.loc[timestamp, 'Close']
                    signal_type = "LONG" if signal == 1 else "SHORT"

                    logged_signals.append({
                        'Timestamp': timestamp,
                        'Signal_Type': signal_type,
                        'Price': price,
                        'Signal_Name': f"RSI_{column}"
                    })

        for column in composite_trade_signals.columns:
            for timestamp, signal in composite_trade_signals[column].items():
                if signal != 0:  # Only log non-zero signals
                    price = self.data.loc[timestamp, 'Close']
                    signal_type = "LONG" if signal == 1 else "SHORT"

                    logged_signals.append({
                        'Timestamp': timestamp,
                        'Signal_Type': signal_type,
                        'Price': price,
                        'Signal_Name': f"Composite_{column}"
                    })

        logged_df = pd.DataFrame(logged_signals)
        logged_df.sort_values('Timestamp', inplace=True)
        downloads_folder = os.path.expanduser("~/Downloads")
        file_path = os.path.join(downloads_folder, filename)
        
        logged_df.to_csv(file_path, index=False)
        print(f"Trade signals logged to {file_path}")

    def optimize_signals(self, signals_to_optimize: List[str]) -> dict:
        optimizer = SignalOptimizer(self, signals_to_optimize)
        return optimizer.optimize_parameters()

if __name__ == "__main__":
    start_date = "2024-09-03"
    end_date = "2024-09-05"
    symbol = "ADAUSDT"
    intervals = ["1m", "15m"]
    
    ALPHA = 0.5
    BETA = 1
    LONG_THRESHOLD = -0.5
    SHORT_THRESHOLD = 0.5

    optimize_signals = True 
    signals_to_optimize = ['RSI14'] 

    for interval in intervals:
        print(f"Processing for interval: {interval}")
        
        processor = TradingSignalProcessor(start_date, end_date)
        processor._fetch_historical_klines(symbol, interval)
        processor.rsi_data = processor._combine_rsi()
        processor.cvi = processor.calculate_cvi()

        if optimize_signals:
            print("Optimizing signals...")
            optimized_params = processor.optimize_signals(signals_to_optimize)
            print("Optimization complete. Optimized parameters:")
            print(optimized_params)

        composite_signals = pd.DataFrame(index=processor.rsi_data.index)
        composite_trade_signals = pd.DataFrame(index=processor.rsi_data.index)

        for column in processor.rsi_data.columns:
            if optimize_signals and column in optimized_params:
                params = optimized_params[column]
                composite_signals[column] = processor.composite_signal(processor.rsi_data[column], processor.cvi, params['ALPHA'], params['BETA'])
                composite_trade_signals[column] = processor.generate_trade_signals(composite_signals[column], params['SHORT_THRESHOLD'], params['LONG_THRESHOLD'])[column]
            else:
                composite_signals[column] = processor.composite_signal(processor.rsi_data[column], processor.cvi, ALPHA, BETA)
                composite_trade_signals[column] = processor.generate_trade_signals(composite_signals[column], SHORT_THRESHOLD, LONG_THRESHOLD)[column]

        rsi_trade_signals = pd.DataFrame(index=processor.rsi_data.index)
        for column in processor.rsi_data.columns:
            rsi_trade_signals[column] = processor.generate_rsi_trade_signals(processor.rsi_data[column])

        composite_signal_quality = processor.analyze_signal_quality(composite_trade_signals)
        composite_trade_counts = processor.get_trade_counts(composite_trade_signals)
        rsi_signal_quality = processor.analyze_signal_quality(rsi_trade_signals)
        rsi_trade_counts = processor.get_trade_counts(rsi_trade_signals)

        output_filename = f"trade_signals_{interval}.csv"
        processor.log_trade_signals(rsi_trade_signals, composite_trade_signals, output_filename)
        processor.plot_signals(processor.rsi_data, processor.cvi, composite_signals, composite_trade_signals)

        print(f"\nResults for interval {interval}:")
        print("Optimized Composite Signals Analysis:")
        print(composite_signal_quality)
        print("\nOptimized Composite Signals Trade Counts:")
        print(composite_trade_counts)
        print("\nRSI Signals Analysis:")
        print(rsi_signal_quality)
        print("\nRSI Signals Trade Counts:")
        print(rsi_trade_counts)
        print("=====================================\n")