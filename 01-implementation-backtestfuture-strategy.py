import asyncio
import websockets
import json
import hmac
import hashlib
import time
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
import talib

API_KEY = 
API_SECRET = 
SYMBOL = 'ADAUSDT'
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LEVERAGE = 10
RISK_PER_TRADE = 0.01
TARGET_PERCENT = 0.02
INTERVAL_STEP = 1

client = Client(API_KEY, API_SECRET)

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
    data['rsi'] = talib.RSI(data['close'], timeperiod=window)
    return data

async def connect_to_binance():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(uri) as websocket:
        return websocket

async def place_order(websocket, side, quantity, order_type="MARKET", stop_price=None, limit_price=None):
    timestamp = int(time.time() * 1000)
    recv_window = 5000

    params = {
        "symbol": SYMBOL,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "recvWindow": recv_window,
        "timestamp": timestamp
    }

    if order_type in ["STOP_LOSS", "TAKE_PROFIT"]:
        params["stopPrice"] = stop_price
    if order_type in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        params["price"] = limit_price

    payload = '&'.join([f'{key}={value}' for key, value in sorted(params.items())])
    signature = hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    logon_message = {
        "id": str(timestamp),
        "method": "order.place",
        "params": params
    }

    await websocket.send(json.dumps(logon_message))
    response = await websocket.recv()
    print(response)

async def get_balance():
    account_info = client.futures_account()
    balance = float(account_info['availableBalance'])
    return balance

async def trading_strategy(websocket):
    balance = await get_balance()
    position = 0
    entry_price = 0
    position_type = None
    price_data = []
    last_trade_index = -INTERVAL_STEP
    prev_rsi = None

    async def handle_trade(trade):
        nonlocal balance, position, entry_price, position_type, price_data, last_trade_index, prev_rsi

        timestamp = trade['T']
        price = float(trade['p'])
        price_data.append(price)
        
        if len(price_data) >= 14:  # Ensure enough data for RSI calculation
            df = pd.DataFrame(price_data, columns=['close'])
            df = calculate_rsi(df, window=14)
            current_rsi = df['rsi'].iloc[-1]

            if position == 0 and len(price_data) > last_trade_index + INTERVAL_STEP:
                if current_rsi <= RSI_OVERSOLD and (prev_rsi is None or prev_rsi > RSI_OVERSOLD):
                    position = (balance * LEVERAGE * RISK_PER_TRADE) / price
                    entry_price = price
                    position_type = 'long'
                    last_trade_index = len(price_data)
                    await place_order(websocket, SIDE_BUY, position)
                    await place_order(websocket, SIDE_SELL, position, order_type="TAKE_PROFIT_LIMIT", limit_price=price * (1 + TARGET_PERCENT))
                    await place_order(websocket, SIDE_SELL, position, order_type="STOP_MARKET", stop_price=price * (1 - (1 / LEVERAGE)))
                    print(f"Opening long position at {price}, balance: {balance}, RSI: {current_rsi}")

                elif current_rsi >= RSI_OVERBOUGHT and (prev_rsi is None or prev_rsi < RSI_OVERBOUGHT):
                    position = (balance * LEVERAGE * RISK_PER_TRADE) / price
                    entry_price = price
                    position_type = 'short'
                    last_trade_index = len(price_data)
                    await place_order(websocket, SIDE_SELL, position)
                    await place_order(websocket, SIDE_BUY, position, order_type="TAKE_PROFIT_LIMIT", limit_price=price * (1 - TARGET_PERCENT))
                    await place_order(websocket, SIDE_BUY, position, order_type="STOP_MARKET", stop_price=price * (1 + (1 / LEVERAGE)))
                    print(f"Opening short position at {price}, balance: {balance}, RSI: {current_rsi}")

            prev_rsi = current_rsi

    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(uri) as websocket:
        while True:
            response = await websocket.recv()
            trade = json.loads(response)
            await handle_trade(trade)

async def fetch_account_info():
    while True:
        balance_info = client.futures_account_balance()
        position_info = client.futures_position_information()
        price_info = client.get_symbol_ticker(symbol=SYMBOL)
        
        balance = balance_info[0]['balance']
        positions = [(pos['symbol'], pos['positionAmt']) for pos in position_info]
        current_price = price_info['price']
        
        print(f"Balance: {balance}, Positions: {positions}, Current Price: {current_price}")
        await asyncio.sleep(60)

async def main():
    websocket = await connect_to_binance()
    await asyncio.gather(
        trading_strategy(websocket),
        fetch_account_info()
    )

asyncio.get_event_loop().run_until_complete(main())
