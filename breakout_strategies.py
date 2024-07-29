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

def calculate_bollinger_bands(df, window=20, num_std=2):
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * num_std)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * num_std)

def generate_signals(df):
    df['signal'] = 0
    df['signal'][df['close'] > df['upper_band']] = 1  # Buy signal
    df['signal'][df['close'] < df['lower_band']] = -1  # Sell signal
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
    plt.plot(df.index, df['rolling_mean'], label='Rolling Mean', alpha=0.7)
    plt.plot(df.index, df['upper_band'], label='Upper Band', linestyle='--', alpha=0.7)
    plt.plot(df.index, df['lower_band'], label='Lower Band', linestyle='--', alpha=0.7)
    plt.scatter(df.index[df['position'] == 1], df['close'][df['position'] == 1], marker='^', color='g', label='Buy Signal', s=100)
    plt.scatter(df.index[df['position'] == -1], df['close'][df['position'] == -1], marker='v', color='r', label='Sell Signal', s=100)
    plt.title('BTC/USDT Breakout Strategy (Bollinger Bands)')
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

# Calculate Bollinger Bands, generate signals, simulate trading, and plot results
calculate_bollinger_bands(df)
generate_signals(df)
final_balance, return_on_investment = simulate_trading(df)
print(f'Final Balance: ${final_balance:.2f}')
print(f'Return on Investment: {return_on_investment:.2f}%')
plot_results(df)