import binance.client
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import plotly.graph_objects as go
import datetime
import pandas as pd
import ta
from binance.client import Client
from binance.enums import FuturesType
import time

# Set up your API key and secret key
api_key = 
api_secret = 
# Initialize the client
client = Client(api_key, api_secret)
while True:
    try:
        # Get futures account balance
        futures_account_info = client.futures_account()

        # Extract relevant balances
        total_wallet_balance = float(futures_account_info['totalWalletBalance'])
        total_unrealized_profit = float(futures_account_info['totalUnrealizedProfit'])
        total_margin_balance = float(futures_account_info['totalMarginBalance'])

        # Get the current price of ADAUSDT
        price_info = client.futures_symbol_ticker(symbol='ADAUSDT')
        price = float(price_info['price'])

        # Print the values
        print(f"Total value (USDT): {total_wallet_balance}")
        print(f"Total unrealized profit (USDT): {total_unrealized_profit}")
        print(f"NET balance (USDT): {total_margin_balance}")
        print(f"ADA price (USDT): {price}")

    except BinanceAPIException as e:
        print(f"Binance API Exception: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Print separator
    print("-----------------------------------------")

    # Wait for 5 seconds before the next iteration
    time.sleep(5)