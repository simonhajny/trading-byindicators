import numpy as np
import pandas as pd
from datetime import datetime
from random import random, uniform, randint

FILE_PATHS = ['/Users/simonlavalle/Downloads/b-data/ADAUSDT_15min_data.csv']
INITIAL_BALANCE = 10000
START_DATE = '2024-03-01'
END_DATE = '2024-08-01'

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class TradingStrategy:
    def __init__(self, data, params, initial_balance=INITIAL_BALANCE):
        self.data = data
        self.params = params
        self.balance = initial_balance
        self.position = None
        self.trades = []

    def generate_signals(self):
        self.data['RSI'] = calculate_rsi(self.data, period=self.params['rsi_period'])
        self.data['Signal'] = np.where(self.data['RSI'] > self.params['rsi_overbought'], -1, 
                                       np.where(self.data['RSI'] < self.params['rsi_oversold'], 1, 0))

    def trade(self):
        for i in range(1, len(self.data)):
            timestamp = self.data['timestamp'].iloc[i]
            signal = self.data['Signal'].iloc[i]
            price = self.data['close'].iloc[i]

            if self.position is None and signal != 0:
                risk_amount = self.balance * self.params['risk_per_trade']
                position_size = risk_amount * self.params['leverage']
                self.position = {
                    'type': 'long' if signal == 1 else 'short',
                    'entry_price': price,
                    'leverage': self.params['leverage'],
                    'size': position_size
                }
                self.trades.append({'type': 'open', 'price': price, 'balance': self.balance, 'position_type': self.position['type'], 'timestamp': timestamp})
            
            elif self.position is not None:
                price_change = (price - self.position['entry_price']) / self.position['entry_price']
                if self.position['type'] == 'short':
                    price_change *= -1

                profit_or_loss = self.position['size'] * price_change
                if price_change >= self.params['target_percent']:
                    self.balance += profit_or_loss
                    self.trades.append({'type': 'close', 'price': price, 'balance': self.balance, 'position_type': self.position['type'], 'timestamp': timestamp})
                    self.position = None

                if self.position is not None: 
                    liquidation_threshold = -(1 / self.position['leverage'])
                    if price_change <= liquidation_threshold:
                        self.balance += profit_or_loss
                        self.trades.append({'type': 'liquidation', 'price': price, 'balance': self.balance, 'position_type': self.position['type'], 'timestamp': timestamp})
                        self.position = None

                        if self.balance < 0:
                            self.balance = 0


class MetropolisOptimization:
    def __init__(self, strategy_class, data, iterations=9000):
        self.strategy_class = strategy_class
        self.data = data
        self.iterations = iterations
        self.best_params = None
        self.best_balance = -np.inf

    def optimize(self):
        for _ in range(self.iterations):
            current_params = {
                'rsi_period': randint(10, 20),
                'rsi_oversold': randint(15, 30),
                'rsi_overbought': randint(70, 85),
                'leverage': uniform(8, 20),
                'risk_per_trade': uniform(0.1, 0.5),
                'target_percent': uniform(0.01, 0.1)
            }

            strategy = self.strategy_class(self.data.copy(), current_params, INITIAL_BALANCE)
            strategy.generate_signals()
            strategy.trade()

            final_balance = strategy.balance
            if final_balance > self.best_balance:
                self.best_balance = final_balance
                self.best_params = current_params

            acceptance_prob = np.exp((final_balance - self.best_balance) / INITIAL_BALANCE)
            if random() < acceptance_prob:
                self.best_balance = final_balance
                self.best_params = current_params

        return self.best_params


class Backtest:
    def __init__(self, strategy_class, data, params):
        self.strategy_class = strategy_class
        self.data = data
        self.params = params

    def run(self):
        strategy = self.strategy_class(self.data.copy(), self.params, INITIAL_BALANCE)
        strategy.generate_signals()
        strategy.trade()
        return strategy.trades, strategy.balance


def load_data(filepath):
    data = pd.read_csv(filepath, parse_dates=['timestamp'])
    data = data[(data['timestamp'] >= START_DATE) & (data['timestamp'] <= END_DATE)]
    return data


if __name__ == "__main__":
    data = load_data(FILE_PATHS[0])
    optimizer = MetropolisOptimization(TradingStrategy, data)
    best_params = optimizer.optimize()
    print(f"Best Parameters: {best_params}")

    backtester = Backtest(TradingStrategy, data, best_params)
    trades, final_balance = backtester.run()
    print(f"Final Balance: {final_balance}")
    for trade in trades:
        print(trade)