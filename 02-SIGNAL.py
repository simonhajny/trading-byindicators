import numpy as np
import requests
import logging
import pandas as pd
from typing import Dict, List, Union, Tuple
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class DataFetcher:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_historical_klines(self, symbol: str, interval: str) -> pd.DataFrame:
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
        return data

class SignalGenerator:
    def __init__(self, data: pd.DataFrame, rsi_windows: List[int] = [10, 14, 21]):
        self.data = data
        self.rsi_windows = rsi_windows

    def calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def normalize_rsi(self, rsi: pd.Series) -> pd.Series:
        return (rsi - 50) / 50

    def combine_rsi(self) -> pd.DataFrame:
        rsi_data = pd.DataFrame(index=self.data.index)
        for window in self.rsi_windows:
            rsi_data[f'RSI_{window}'] = self.calculate_rsi(self.data['Close'], window)

        for column in rsi_data.columns:
            rsi_data[column] = self.normalize_rsi(rsi_data[column])
        rsi_data.columns = [f'RSI{w}' for w in self.rsi_windows]
        rsi_data['RSI_S'] = rsi_data.mean(axis=1)
        weights = np.array(range(len(self.rsi_windows), 0, -1))
        rsi_data['RSI_W'] = np.average(rsi_data[[f'RSI{w}' for w in self.rsi_windows]], axis=1, weights=weights)
        exp_weights = np.exp(range(len(self.rsi_windows)))
        rsi_data['RSI_E'] = np.average(rsi_data[[f'RSI{w}' for w in self.rsi_windows]], axis=1, weights=exp_weights)
        
        return rsi_data

    def calculate_cvi(self, ema_window: int = 10, roc_window: int = 10) -> pd.Series:
        high_low_diff = self.data['High'] - self.data['Low']
        ema_diff = high_low_diff.ewm(span=ema_window, adjust=False).mean()
        cvi = ((ema_diff - ema_diff.shift(roc_window)) / ema_diff.shift(roc_window)) * 100
        return cvi

    def composite_signal(self, rsi_norm: pd.Series, cvi: pd.Series, alpha: float, beta: float) -> pd.Series:
        cvi_pos_norm = np.minimum(cvi / 100, 1)
        cvi_neg_norm = np.maximum(cvi / 100, -1)
        cvi_factor = np.where(
            cvi >= 0,
            1 + alpha * np.power(cvi_pos_norm, 2),
            1 / (1 + beta * np.abs(cvi_neg_norm))
        )
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
                    signals.loc[signal.index[i], column] = 0  # No signal
        return signals

    def generate_rsi_trade_signals(self, rsi_norm: pd.Series) -> pd.Series:
        return self.generate_trade_signals(rsi_norm, 0.75, -0.75)

class SignalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def analyze_signal_quality(self, signals: pd.DataFrame, target_pct: float = 0.005, stop_loss_pct: float = 0.01, max_holding_periods: int = 24) -> pd.DataFrame:
        signal_quality = pd.DataFrame(index=signals.columns,
                                    columns=['Total_Signals', 'Win_Rate', 'Avg_Win_Pct', 'Avg_Loss_Pct', 'Profit_Factor'])
        price = self.data['Close']
        
        for column in signals.columns:
            long_signals = signals[column] == 1
            short_signals = signals[column] == -1
            total_signals = long_signals.sum() + short_signals.sum()
            wins = 0
            losses = 0
            total_win_pct = 0
            total_loss_pct = 0
            
            for signal_time in price[long_signals].index:
                future_prices = price.loc[signal_time:signal_time + pd.Timedelta(hours=max_holding_periods)]
                if len(future_prices) > 1:
                    returns = (future_prices / future_prices.iloc[0] - 1)
                    max_return = returns.max()
                    max_drawdown = returns.min()
                    
                    if max_return >= target_pct:
                        wins += 1
                        total_win_pct += max_return
                    elif max_drawdown <= -stop_loss_pct:
                        losses += 1
                        total_loss_pct += max_drawdown
                    else:
                        final_return = returns.iloc[-1]
                        if final_return >= 0:
                            wins += 1
                            total_win_pct += final_return
                        else:
                            losses += 1
                            total_loss_pct += final_return
            
            for signal_time in price[short_signals].index:
                future_prices = price.loc[signal_time:signal_time + pd.Timedelta(hours=max_holding_periods)]
                if len(future_prices) > 1:
                    returns = -(future_prices / future_prices.iloc[0] - 1)  # Negative for short positions
                    max_return = returns.max()
                    max_drawdown = returns.min()
                    
                    if max_return >= target_pct:
                        wins += 1
                        total_win_pct += max_return
                    elif max_drawdown <= -stop_loss_pct:
                        losses += 1
                        total_loss_pct += max_drawdown
                    else:
                        # If neither target nor stop-loss is hit, consider the last return
                        final_return = returns.iloc[-1]
                        if final_return >= 0:
                            wins += 1
                            total_win_pct += final_return
                        else:
                            losses += 1
                            total_loss_pct += final_return
            
            signal_quality.loc[column, 'Total_Signals'] = total_signals
            signal_quality.loc[column, 'Win_Rate'] = wins / total_signals if total_signals > 0 else 0
            signal_quality.loc[column, 'Avg_Win_Pct'] = total_win_pct / wins if wins > 0 else 0
            signal_quality.loc[column, 'Avg_Loss_Pct'] = total_loss_pct / losses if losses > 0 else 0
            signal_quality.loc[column, 'Profit_Factor'] = (total_win_pct / -total_loss_pct) if total_loss_pct != 0 else float('inf')
        
        return signal_quality

    def get_trade_counts(self, signals: pd.DataFrame) -> pd.DataFrame:
        trade_counts = pd.DataFrame(index=signals.columns, columns=['Long_Trades', 'Short_Trades'])
        for column in signals.columns:
            trade_counts.loc[column, 'Long_Trades'] = (signals[column] == 1).sum()
            trade_counts.loc[column, 'Short_Trades'] = (signals[column] == -1).sum()
        return trade_counts

class SignalOptimizer:
    def __init__(self, data: pd.DataFrame, rsi_data: pd.DataFrame, cvi: pd.Series):
        self.data = data
        self.rsi_data = rsi_data
        self.cvi = cvi
        self.signal_generator = SignalGenerator(data)
        self.signal_analyzer = SignalAnalyzer(data)

    def optimize_signals(self) -> dict:
        optimized_params = {}
        composite_signals = [f"Composite_{col}" for col in self.rsi_data.columns]
        
        def objective(params, signal):
            alpha, beta, long_threshold, short_threshold = params
            rsi_column = signal.replace("Composite_", "")
            composite_signal = self.signal_generator.composite_signal(self.rsi_data[rsi_column], self.cvi, alpha, beta)
            trade_signals = self.signal_generator.generate_trade_signals(pd.DataFrame({signal: composite_signal}), short_threshold, long_threshold)
            signal_quality = self.signal_analyzer.analyze_signal_quality(trade_signals)
            win_rate = signal_quality.loc[signal, 'Win_Rate']
            total_signals = signal_quality.loc[signal, 'Total_Signals']
            signal_penalty = 1 - np.exp(-total_signals / 100)
            
            return -win_rate * signal_penalty

        space = [
            Real(0, 7, name='alpha'),
            Real(0, 7, name='beta'),
            Real(-0.8, -0.5, name='long_threshold'),
            Real(0.5, 0.8, name='short_threshold')
        ]

        for signal in composite_signals:
            @use_named_args(space)
            def wrapped_objective(**kwargs):
                return objective([kwargs['alpha'], kwargs['beta'], kwargs['long_threshold'], kwargs['short_threshold']], signal)

            result = gp_minimize(wrapped_objective, space, n_calls=50, random_state=42, n_jobs=-1)
            optimized_params[signal] = {
                'ALPHA': result.x[0],
                'BETA': result.x[1],
                'LONG_THRESHOLD': result.x[2],
                'SHORT_THRESHOLD': result.x[3],
                'BEST_SCORE': -result.fun  # Store the best score
            }

        return optimized_params


class SignalVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_signals(self, rsi_data: pd.DataFrame, cvi: pd.Series, composite_signals: pd.DataFrame, trade_signals: Union[pd.Series, pd.DataFrame]):
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=["Price", "Normalized RSIs", "CVI", "Composite Signals", "Trade Signals"])

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06EC4']
        fig.add_trace(go.Candlestick(x=self.data.index, open=self.data['Open'], high=self.data['High'], 
                                    low=self.data['Low'], close=self.data['Close'], name='Price'), row=1, col=1)

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

        for i, column in enumerate(rsi_data.columns):
            fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data[column], mode='lines', 
                                    name=column, line=dict(color=colors[i % len(colors)], width=1)), row=2, col=1)
        
        fig.add_hline(y=-0.5, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
        fig.add_hline(y=0.5, line=dict(color='red', width=1, dash='dash'), row=2, col=1)

        fig.add_trace(go.Scatter(x=cvi.index, y=cvi, mode='lines', name='CVI', 
                                line=dict(color='white', width=1)), row=3, col=1)

        for i, column in enumerate(composite_signals.columns):
            fig.add_trace(go.Scatter(x=composite_signals.index, y=composite_signals[column], mode='lines', 
                                    name=f'Comp_{column}', line=dict(color=colors[i % len(colors)], width=1)), row=4, col=1)
        
        fig.add_hline(y=-0.5, line=dict(color='red', width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=0.5, line=dict(color='red', width=1, dash='dash'), row=4, col=1)

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

class TradingSignalProcessor:
    def __init__(self, start_date: str, end_date: str, symbol: str, interval: str, rsi_windows: List[int] = [10, 14, 21]):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.interval = interval
        self.rsi_windows = rsi_windows
        self.data_fetcher = DataFetcher(start_date, end_date)
        self.data = None
        self.signal_generator = None
        self.signal_analyzer = None
        self.signal_optimizer = None
        self.signal_visualizer = None
        self.rsi_data = None
        self.cvi = None
        self.composite_signals = None
        self.composite_trade_signals = None
        self.rsi_trade_signals = None
        self.optimized_params = None

    def fetch_data(self):
        self.data = self.data_fetcher.fetch_historical_klines(self.symbol, self.interval)
        self.signal_generator = SignalGenerator(self.data, self.rsi_windows)
        self.signal_analyzer = SignalAnalyzer(self.data)
        self.signal_visualizer = SignalVisualizer(self.data)

    def generate_signals(self):
        self.rsi_data = self.signal_generator.combine_rsi()
        self.cvi = self.signal_generator.calculate_cvi()
        self.signal_optimizer = SignalOptimizer(self.data, self.rsi_data, self.cvi)

    def optimize_signals(self):
        self.optimized_params = self.signal_optimizer.optimize_signals()

    def generate_trade_signals(self):
        self.composite_signals = pd.DataFrame(index=self.rsi_data.index)
        self.composite_trade_signals = pd.DataFrame(index=self.rsi_data.index)
        self.rsi_trade_signals = pd.DataFrame(index=self.rsi_data.index)

        for rsi_column in self.rsi_data.columns:
            composite_column = f"Composite_{rsi_column}"
            params = self.optimized_params[composite_column]
            self.composite_signals[composite_column] = self.signal_generator.composite_signal(
                self.rsi_data[rsi_column], self.cvi, 
                params['ALPHA'], params['BETA']
            )
            self.composite_trade_signals[composite_column] = self.signal_generator.generate_trade_signals(
                self.composite_signals[composite_column], 
                params['SHORT_THRESHOLD'], params['LONG_THRESHOLD']
            )
            self.rsi_trade_signals[rsi_column] = self.signal_generator.generate_rsi_trade_signals(self.rsi_data[rsi_column])

    def analyze_signals(self):
        composite_trade_counts = self.signal_analyzer.get_trade_counts(self.composite_trade_signals)
        rsi_trade_counts = self.signal_analyzer.get_trade_counts(self.rsi_trade_signals)
        composite_signal_quality = self.signal_analyzer.analyze_signal_quality(self.composite_trade_signals)
        rsi_signal_quality = self.signal_analyzer.analyze_signal_quality(self.rsi_trade_signals)
        
        return composite_trade_counts, rsi_trade_counts, composite_signal_quality, rsi_signal_quality

    def visualize_signals(self):
        self.signal_visualizer.plot_signals(self.rsi_data, self.cvi, self.composite_signals, self.composite_trade_signals)

    def log_trade_signals(self, filename: str = "trade_signals.csv"):
        logged_signals = []

        for column in self.rsi_trade_signals.columns:
            for timestamp, signal in self.rsi_trade_signals[column].items():
                if signal != 0:  # Only log non-zero signals
                    price = self.data.loc[timestamp, 'Close']
                    signal_type = "LONG" if signal == 1 else "SHORT"

                    logged_signals.append({
                        'Timestamp': timestamp,
                        'Signal_Type': signal_type,
                        'Price': price,
                        'Signal_Name': f"{column}"
                    })

        for column in self.composite_trade_signals.columns:
            for timestamp, signal in self.composite_trade_signals[column].items():
                if signal != 0:  # Only log non-zero signals
                    price = self.data.loc[timestamp, 'Close']
                    signal_type = "LONG" if signal == 1 else "SHORT"

                    logged_signals.append({
                        'Timestamp': timestamp,
                        'Signal_Type': signal_type,
                        'Price': price,
                        'Signal_Name': f"{column}"
                    })

        logged_df = pd.DataFrame(logged_signals)
        logged_df.sort_values('Timestamp', inplace=True)
        downloads_folder = os.path.expanduser("~/Downloads")
        file_path = os.path.join(downloads_folder, filename)
        logged_df.to_csv(file_path, index=False)
        print(f"Trade signals logged to {file_path}")

    def optimize_signals(self):
        self.optimized_params = self.signal_optimizer.optimize_signals()
        print("\nOptimized Parameters:")
        for signal, params in self.optimized_params.items():
            print(f"{signal}:")
            for param, value in params.items():
                print(f"  {param}: {value:.4f}")

    def process(self):
        self.fetch_data()
        self.generate_signals()
        self.optimize_signals()
        self.generate_trade_signals()
        composite_trade_counts, rsi_trade_counts, composite_signal_quality, rsi_signal_quality = self.analyze_signals()
        self.visualize_signals()
        self.log_trade_signals(f"trade_signals_{self.interval}.csv")

        print(f"\nResults for interval {self.interval}:")
        print("Optimized Parameters:")
        for signal, params in self.optimized_params.items():
            print(f"{signal}:")
            for param, value in params.items():
                print(f"  {param}: {value:.4f}")
        print("\nIndividualized Composite Signals Analysis:")
        print(composite_signal_quality)
        print("\nIndividualized Composite Signals Trade Counts:")
        print(composite_trade_counts)
        print("\nRSI Signals Analysis:")
        print(rsi_signal_quality)
        print("\nRSI Signals Trade Counts:")
        print(rsi_trade_counts)
        print("=====================================\n")

if __name__ == "__main__":
    start_date = "2024-09-09"
    end_date = "2024-09-10"
    symbol = "ADAUSDT"
    intervals = ["1m"]

    for interval in intervals:
        print(f"Processing for interval: {interval}")
        processor = TradingSignalProcessor(start_date, end_date, symbol, interval)
        processor.process()