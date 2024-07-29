import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_prices():
    url = "https://api.binance.com/api/v3/ticker/price"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df.set_index('symbol', inplace=True)
    df['price'] = df['price'].astype(float)
    return df

def find_arbitrage_opportunity(df):
    pairs = ['BTCUSDT', 'ETHBTC', 'ETHUSDT']
    
    if not all(pair in df.index for pair in pairs):
        raise ValueError("Required trading pairs not found in the data.")

    btc_usdt = df.loc['BTCUSDT', 'price']
    eth_btc = df.loc['ETHBTC', 'price']
    eth_usdt = df.loc['ETHUSDT', 'price']

    # Calculate the arbitrage profit
    start_amount = 1  # Start with 1 BTC
    eth_amount = start_amount * eth_btc
    usdt_amount = eth_amount * eth_usdt
    profit = usdt_amount - start_amount * btc_usdt

    return profit

def simulate_arbitrage(initial_balance=1):
    df = fetch_prices()
    profit = find_arbitrage_opportunity(df)
    final_balance = initial_balance + profit
    return_on_investment = (profit / initial_balance) * 100
    return final_balance, return_on_investment

def plot_results():
    initial_balance = 1  # Start with 1 BTC
    final_balance, return_on_investment = simulate_arbitrage(initial_balance)
    
    print(f'Final Balance: ${final_balance:.2f}')
    print(f'Return on Investment: {return_on_investment:.2f}%')

# Main execution
plot_results()