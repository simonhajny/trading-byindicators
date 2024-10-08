import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from ta.momentum import RSIIndicator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, exchange: ccxt.Exchange, symbol: str, timeframe: str, buffer_size: int = 100):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        self.cached_data = pd.DataFrame()

    def fetch_ohlcv(self) -> pd.DataFrame:
        try:
            logger.debug(f"Fetching OHLCV data for {self.symbol}")
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                limit=self.buffer_size + 1  # Fetch an extra candle to ensure we have enough data
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)  # Set utc=True here
            df.set_index('timestamp', inplace=True)
            now = datetime.now(timezone.utc)
            last_closed_time = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
            df = df[df.index <= last_closed_time]
            df = df.tail(self.buffer_size)
            self.cached_data = df
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return self.cached_data

    def get_latest_price(self) -> float:
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            return np.nan

class SignalGenerator:
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.data_buffer = pd.DataFrame()
        self.rsi_windows = [10, 14, 21]

    def update_data(self):
        new_data = self.data_fetcher.fetch_ohlcv()
        if self.data_buffer.empty:
            self.data_buffer = new_data
        else:
            self.data_buffer = pd.concat([self.data_buffer, new_data])             # Concatenate new data and drop duplicates
            self.data_buffer = self.data_buffer[~self.data_buffer.index.duplicated(keep='last')]            # Keep only the latest 'buffer_size' data points
            self.data_buffer = self.data_buffer.tail(self.data_fetcher.buffer_size)

    def calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        rsi_indicator = RSIIndicator(close=prices, window=window)
        return rsi_indicator.rsi()

    def normalize_rsi(self, rsi: pd.Series) -> pd.Series:
        return (rsi - 50) / 50

    def combine_rsi(self) -> pd.DataFrame:
        rsi_data = pd.DataFrame(index=self.data_buffer.index)
        for window in self.rsi_windows:
            rsi = self.calculate_rsi(self.data_buffer['Close'], window)
            rsi_norm = self.normalize_rsi(rsi)
            rsi_data[f'RSI_{window}'] = rsi_norm

        rsi_columns = [f'RSI_{w}' for w in self.rsi_windows]
        rsi_data['RSI_S'] = rsi_data[rsi_columns].mean(axis=1)

        weights = np.array(range(len(self.rsi_windows), 0, -1))
        rsi_data['RSI_W'] = rsi_data[rsi_columns].apply(
            lambda x: np.average(x, weights=weights), axis=1
        )

        exp_weights = np.exp(range(len(self.rsi_windows)))
        rsi_data['RSI_E'] = rsi_data[rsi_columns].apply(
            lambda x: np.average(x, weights=exp_weights), axis=1
        )

        return rsi_data

    def generate_signals(self, alpha: float, beta: float) -> Dict[str, float]:
        self.update_data()
        if self.data_buffer.empty:
            logger.warning("Data buffer is empty. Cannot generate signals.")
            return {}

        rsi_data = self.combine_rsi()
        signals = {}
        for column in rsi_data.columns:
            signals[column] = rsi_data[column].iloc[-1]

        return signals

class TradingStrategy:
    def __init__(self, signals: Dict[str, float], selected_signal: str, long_threshold: float, short_threshold: float):
        self.signals = signals
        self.selected_signal = selected_signal
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def evaluate(self) -> int:
        signal_value = self.signals.get(self.selected_signal, None)
        if signal_value is None or np.isnan(signal_value):
            logger.warning(f"Signal '{self.selected_signal}' not available or NaN.")
            return 0  

        logger.info(f"Signal '{self.selected_signal}' Value: {signal_value:.4f}")

        if signal_value <= self.long_threshold:
            logger.info(
                f"Signal '{self.selected_signal}' crossed below long threshold ({self.long_threshold}). Generating LONG signal."
            )
            return 1 
        elif signal_value >= self.short_threshold:
            logger.info(
                f"Signal '{self.selected_signal}' crossed above short threshold ({self.short_threshold}). Generating SHORT signal."
            )
            return -1 
        else:
            logger.info(f"Signal '{self.selected_signal}' is between thresholds. No action taken.")
            return 0  

class PositionManager:
    def __init__(self, exchange: ccxt.Exchange, symbol: str, leverage: float, risk_percent: float, target_percent: float, stop_loss_percent: float):
        self.exchange = exchange
        self.symbol = symbol
        self.leverage = leverage
        self.risk_percent = risk_percent
        self.target_percent = target_percent
        self.stop_loss_percent = stop_loss_percent
        self.position = None
        self.futures_balance = None
        self.set_leverage()

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance(params={"type": "future"})
            self.futures_balance = balance['USDT']
            return self.futures_balance['free']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def set_leverage(self):
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")

    def get_open_position(self):
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            position = next((p for p in positions if p['symbol'] == self.symbol and float(p['contracts']) != 0), None)
            if position:
                self.position = {
                    'side': 'long' if float(position['contracts']) > 0 else 'short',
                    'amount': abs(float(position['contracts'])),
                    'entry_price': float(position['entryPrice']),
                    'unrealized_pnl': float(position['unrealizedPnl']),
                    'leverage': float(position['leverage']),
                    'liquidation_price': float(position['liquidationPrice']) if 'liquidationPrice' in position else None,
                }
                logger.info(f"Current open position: {self.position}")
            else:
                self.position = None
                logger.info("No open position found")
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            self.position = None

    def calculate_order_size(self, price: float) -> float:
        balance = self.get_balance()
        risk_amount = balance * self.risk_percent
        order_size = (risk_amount * self.leverage) / price
        return order_size

    def open_position(self, side: str, amount: float):
        try:
            order_side = 'buy' if side == 'buy' else 'sell'
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='MARKET',
                side=order_side,
                amount=amount,
                params={'type': 'future'}
            )

            price = self.exchange.fetch_ticker(self.symbol)['last']
            if side == 'buy':
                stop_loss_price = price * (1 - self.stop_loss_percent)
                take_profit_price = price * (1 + self.target_percent)
            else:
                stop_loss_price = price * (1 + self.stop_loss_percent)
                take_profit_price = price * (1 - self.target_percent)

            self.exchange.create_order(
                symbol=self.symbol,
                type='STOP_MARKET',
                side='sell' if side == 'buy' else 'buy',
                amount=amount,
                params={'stopPrice': stop_loss_price}
            )

            self.exchange.create_order(
                symbol=self.symbol,
                type='TAKE_PROFIT_MARKET',
                side='sell' if side == 'buy' else 'buy',
                amount=amount,
                params={'stopPrice': take_profit_price}
            )

            self.position = {'side': side, 'amount': amount, 'entry_price': price}

            logger.info(f"Opened {side.upper()} position: Amount={amount}, Entry Price={price}")
            logger.info(f"Set Stop Loss at {stop_loss_price:.4f} and Take Profit at {take_profit_price:.4f}")

        except Exception as e:
            logger.error(f"Error opening position: {e}")

    def close_position(self):
        if self.position:
            try:
                side = 'sell' if self.position['side'] == 'buy' else 'buy'
                amount = abs(float(self.position['amount']))
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='MARKET',
                    side=side,
                    amount=amount,
                    params={'type': 'future'}
                )
                logger.info(f"Closed position: {order}")
                self.position = None
            except Exception as e:
                logger.error(f"Error closing position: {e}")

    def update_position(self, side: str):
        if side == 'neutral':
            if self.position:
                self.close_position()
            else:
                logger.info("No position to close for neutral signal.")
            return

        if not self.position:
            price = self.exchange.fetch_ticker(self.symbol)['last']
            amount = self.calculate_order_size(price)
            self.open_position(side, amount)
        elif self.position['side'] != side:
            self.close_position()
            price = self.exchange.fetch_ticker(self.symbol)['last']
            amount = self.calculate_order_size(price)
            self.open_position(side, amount)
        else:
            logger.info("Position already open in the same direction. No action taken.")

class TradingBot:
    def __init__(self, config: Dict):
        self.exchange = ccxt.binance({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        self.symbol = config['symbol']
        self.timeframe = config['timeframe']
        self.leverage = config['leverage']
        self.risk_percent = config['risk_percent']
        self.long_threshold = config['long_threshold']
        self.short_threshold = config['short_threshold']
        self.selected_signal = config['selected_signal']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.target_percent = config['target_percent']
        self.stop_loss_percent = config['stop_loss_percent']

        self.data_fetcher = DataFetcher(self.exchange, self.symbol, self.timeframe)
        self.signal_generator = SignalGenerator(self.data_fetcher)
        self.position_manager = PositionManager(
            self.exchange,
            self.symbol,
            self.leverage,
            self.risk_percent,
            self.target_percent,
            self.stop_loss_percent
        )

        self.previous_trade_signal: Optional[int] = None  # Keep track of the previous trade signal

    def run(self):
        while True:
            try:
                time.sleep(10)
                signals = self.signal_generator.generate_signals(self.alpha, self.beta)
                if not signals:
                    logger.warning("No signals generated.")
                    continue
                strategy = TradingStrategy(
                    signals,
                    self.selected_signal,
                    self.long_threshold,
                    self.short_threshold
                )
                trade_signal = strategy.evaluate()
                self.position_manager.get_open_position()
                self.log_status(signals, trade_signal)
                if self.previous_trade_signal != trade_signal:
                    if trade_signal == 1:
                        side = 'buy'
                    elif trade_signal == -1:
                        side = 'sell'
                    else:
                        side = 'neutral'

                    self.position_manager.update_position(side)
                    self.previous_trade_signal = trade_signal
                else:
                    logger.info("No signal crossing detected. No position update required.")

            except Exception as e:
                logger.error(f"Error in bot loop: {e}")
                time.sleep(60)

    def log_status(self, signals: Dict[str, float], trade_signal: int):
        try:
            current_price = self.data_fetcher.get_latest_price()
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            current_price = "Unknown"

        signal_value = signals.get(self.selected_signal, 'N/A')
        signal_direction = {1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'}
        trade_direction = signal_direction.get(trade_signal, 'UNKNOWN')
        self.position_manager.get_balance()
        self.position_manager.get_open_position()
        log_message = (
            f"\n{'='*60}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"{'-'*60}\n"
        )
        log_message += (
            f"Current Price      : {current_price:.4f}\n"
            f"Selected Signal    : {self.selected_signal}\n"
            f"Signal Value       : {signal_value:.4f}\n"
            f"Trade Signal       : {trade_direction}\n"
            f"{'-'*60}\n"
        )

        log_message += "Parameters:\n"
        log_message += (
            f"  Long Threshold    : {self.long_threshold}\n"
            f"  Short Threshold   : {self.short_threshold}\n"
            f"  Leverage          : {self.leverage}\n"
            f"  Risk per Trade    : {self.risk_percent}\n"
            f"  Target Percent    : {self.target_percent}\n"
            f"  Stop Loss Percent : {self.stop_loss_percent}\n"
            f"  Alpha             : {self.alpha}\n"
            f"  Beta              : {self.beta}\n"
            f"{'-'*60}\n"
        )

        if self.position_manager.futures_balance:
            fb = self.position_manager.futures_balance
            log_message += "Futures Account Balance:\n"
            log_message += (
                f"  Free   : {fb['free']} USDT\n"
                f"  Used   : {fb['used']} USDT\n"
                f"  Total  : {fb['total']} USDT\n"
                f"{'-'*60}\n"
            )

        if self.position_manager.position:
            position = self.position_manager.position
            log_message += "Active Position:\n"
            log_message += (
                f"  Side               : {position['side']}\n"
                f"  Amount             : {position['amount']}\n"
                f"  Entry Price        : {position['entry_price']}\n"
                f"  Unrealized PNL     : {position['unrealized_pnl']} USDT\n"
                f"  Leverage           : {position['leverage']}x\n"
            )
            if position['liquidation_price']:
                log_message += f"  Liquidation Price  : {position['liquidation_price']}\n"
            log_message += f"{'='*60}\n"
        else:
            log_message += "No active position\n"
            log_message += f"{'='*60}\n"

        logger.info(log_message)

if __name__ == "__main__":
    config = {
        'api_key':'',
        'api_secret':'',
        'symbol': 'ADA/USDT',
        'timeframe': '1m',
        'leverage': 30,
        'risk_percent': 0.01,
        'target_percent': 0.004,  
        'stop_loss_percent': 0.005,  
        'long_threshold': -0.45,
        'short_threshold': 0.45,
        'selected_signal': 'RSI_14', 
        'alpha': 0.5,
        'beta': 0.5,
    }

    bot = TradingBot(config)
    bot.run()

