import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def get_current_endtime():
    return int(time.time() * 1000)

def create_url(symbol='BTCUSDT', interval='1d', limit=1000):
    end_time = get_current_endtime()
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"

def fetch_data(url):
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

def generate_signals(df, rsi_overbought=70, rsi_oversold=30):
    df['signal'] = 0
    df['signal'][df['rsi'] < rsi_oversold] = 1  # Buy signal
    df['signal'][df['rsi'] > rsi_overbought] = -1  # Sell signal
    df['position'] = df['signal'].diff()

def simulate_trading(df, initial_balance=1000):
    balance = initial_balance
    position = 0
    buy_price = 0

    for i in range(len(df)):
        if df['position'].iloc[i] == 1:  # Buy signal
            if position == 0:  # If not already in position
                buy_price = df['close'].iloc[i]
                position = balance / buy_price
                balance = 0
        elif df['position'].iloc[i] == -1:  # Sell signal
            if position > 0:  # If in position
                balance = position * df['close'].iloc[i]
                position = 0

    # Calculate final portfolio value
    if position > 0:  # If still holding BTC
        balance = position * df['close'].iloc[-1]

    final_balance = balance
    return_on_investment = (final_balance - initial_balance) / initial_balance * 100

    return final_balance, return_on_investment

def plot_results(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='BTC/USDT Price')
    plt.plot(df.index, df['rsi'], label='RSI', alpha=0.7)
    plt.axhline(y=70, color='r', linestyle='--', label='RSI Overbought')
    plt.axhline(y=30, color='g', linestyle='--', label='RSI Oversold')
    plt.scatter(df.index[df['position'] == 1], df['close'][df['position'] == 1], marker='^', color='g', label='Buy Signal', s=100)
    plt.scatter(df.index[df['position'] == -1], df['close'][df['position'] == -1], marker='v', color='r', label='Sell Signal', s=100)
    plt.title('BTC/USDT Momentum Trading Strategy (RSI)')
    plt.legend()
    plt.show()

# Main execution
url = create_url()
df = fetch_data(url)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Convert columns to numeric
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# Calculate RSI, generate signals, simulate trading, and plot results
calculate_rsi(df)
generate_signals(df)
final_balance, return_on_investment = simulate_trading(df)
print(f'Final Balance: ${final_balance:.2f}')
print(f'Return on Investment: {return_on_investment:.2f}%')
plot_results(df)