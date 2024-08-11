import requests
import time
import hmac
import hashlib
import logging
import pandas as pd
import talib

API_KEY = ''
API_SECRET = ''
BASE_URL = 'https://fapi.binance.com' 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sign_request(params):
    """Signs the request parameters using the API secret."""
    query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + '&signature=' + signature

def send_signed_request(http_method, endpoint, payload=None):
    """Sends a signed request to the Binance Futures API."""
    url = BASE_URL + endpoint
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    if payload is None:
        payload = {}
    payload['recvWindow'] = 5000
    payload['timestamp'] = int(time.time() * 1000)
    
    signed_payload = sign_request(payload)
    if http_method == 'GET':
        response = requests.get(url, headers=headers, params=signed_payload)
    elif http_method == 'POST':
        response = requests.post(url, headers=headers, params=signed_payload)
    else:
        raise ValueError("Unsupported HTTP method")

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred: {err}")
        return None
    except Exception as err:
        logging.error(f"Other error occurred: {err}")
        return None

    return response.json()

def place_order(symbol, side, order_type, quantity, stop_price=None, activate_price=None, price_rate=None, time_in_force='GTC'):
    """Places an order on the Binance Futures API."""
    payload = {
        'symbol': symbol,
        'side': side,
        'type': order_type,
        'quantity': quantity,
        'timeInForce': time_in_force,
    }
    if stop_price:
        payload['stopPrice'] = stop_price
    if activate_price:
        payload['activatePrice'] = activate_price
    if price_rate:
        payload['priceRate'] = price_rate

    logging.info(f"Placing order: {payload}")
    return send_signed_request('POST', '/fapi/v1/order', payload)

def get_order_status(symbol, order_id):
    """Gets the status of an order from the Binance Futures API."""
    payload = {
        'symbol': symbol,
        'orderId': order_id,
    }
    
    logging.info(f"Getting order status: Symbol={symbol}, Order ID={order_id}")
    return send_signed_request('GET', '/fapi/v1/order', payload)

def get_account_balance():
    """Fetches the account balance from the Binance Futures API."""
    logging.info("Fetching account balance")
    return send_signed_request('GET', '/fapi/v3/account')

def get_candlestick_data(symbol, interval, limit=100):
    url = BASE_URL + '/fapi/v1/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = df['close'].astype(float)
        return df
    return pd.DataFrame()

def calculate_rsi(data, window=14):
    data['rsi'] = talib.RSI(data['close'], timeperiod=window)
    return data

def trading_logic(interval):
    symbol = 'ADAUSDT'
    balance = get_account_balance() #availableBalance': '0.00460608'
    position = 0
    position_type = None
    entry_price = 0
    last_trade_index = 0
    trades = []
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    LEVERAGE = 10
    RISK_PER_TRADE = 0.02
    interval_step = 2
    TARGET_PERCENT = 0.01
    prev_rsi = 0

    while True:
        data = get_candlestick_data(symbol, interval)
        data_with_rsi = calculate_rsi(data.copy())

        for current_index, (index, row) in enumerate(data_with_rsi.iterrows()):
            current_rsi = row['rsi']

            if position == 0 and current_index >= last_trade_index + interval_step:
                if current_rsi <= RSI_OVERSOLD and prev_rsi > RSI_OVERSOLD:
                    position = (balance * LEVERAGE * RISK_PER_TRADE) / row['close']
                    entry_price = row['close']
                    position_type = 'long'
                    last_trade_index = current_index
                    trades.append(['Opening long', row['close'], index, balance, current_rsi, ''])

                    # Place long order
                    order_response = place_order(
                        symbol=symbol,
                        side='BUY',
                        order_type='TRAILING_STOP_MARKET',
                        quantity=position,
                        stop_price=entry_price * (1 - (1 / LEVERAGE)),
                        activate_price=entry_price,
                        price_rate=TARGET_PERCENT
                    )
                    print(f"Order Response: {order_response}")

                elif current_rsi >= RSI_OVERBOUGHT and prev_rsi < RSI_OVERBOUGHT:
                    position = (balance * LEVERAGE * RISK_PER_TRADE) / row['close']
                    entry_price = row['close']
                    position_type = 'short'
                    last_trade_index = current_index
                    trades.append(['Opening short', row['close'], index, balance, current_rsi, ''])

                    # Place short order
                    order_response = place_order(
                        symbol=symbol,
                        side='SELL',
                        order_type='TRAILING_STOP_MARKET',
                        quantity=position,
                        stop_price=entry_price * (1 + (1 / LEVERAGE)),
                        activate_price=entry_price,
                        price_rate=TARGET_PERCENT
                    )
                    print(f"Order Response: {order_response}")

            elif position != 0:
                if position_type == 'long':
                    if row['close'] >= entry_price * (1 + TARGET_PERCENT):
                        profit = position * (row['close'] - entry_price)
                        balance += profit
                        trades.append(['Closing long', row['close'], index, balance, current_rsi, profit])
                        position = 0

                    elif row['close'] <= entry_price * (1 - (1 / LEVERAGE)):
                        loss = position * (entry_price - row['close'])
                        balance -= loss
                        trades.append(['Liquidating long', row['close'], index, balance, current_rsi, loss])
                        position = 0

                elif position_type == 'short':
                    if row['close'] <= entry_price * (1 - TARGET_PERCENT):
                        profit = position * (entry_price - row['close'])
                        balance += profit
                        trades.append(['Closing short', row['close'], index, balance, current_rsi, profit])
                        position = 0

                    elif row['close'] >= entry_price * (1 + (1 / LEVERAGE)):
                        loss = position * (row['close'] - entry_price)
                        balance -= loss
                        trades.append(['Liquidating short', row['close'], index, balance, current_rsi, loss])
                        position = 0
        prev_rsi = current_rsi
        time.sleep(60) 

if __name__ == "__main__":
    interval = '1m'
    trading_logic(interval)

