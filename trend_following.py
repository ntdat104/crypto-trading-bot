import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def get_current_endtime():
    """Returns current time in milliseconds."""
    return int(time.time() * 1000)

def create_url(symbol='BTCUSDT', interval='1d', limit=1000):
    """Creates a URL to fetch historical data from Binance."""
    end_time = get_current_endtime()
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"

def fetch_data(url):
    """Fetches data from the given URL and returns it as a DataFrame."""
    response = requests.get(url)
    response.raise_for_status()  # Check for request errors
    data = response.json()
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

def preprocess_data(df):
    """Preprocesses the DataFrame by converting timestamps and columns to numeric types."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def calculate_moving_averages(df, short_window=50, long_window=200):
    """Calculates short-term and long-term moving averages."""
    df['SMA_short'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    return df

def generate_signals(df, short_window=50):
    """Generates buy/sell signals based on moving average crossovers."""
    df['signal'] = 0
    df['signal'][short_window:] = np.where(df['SMA_short'][short_window:] > df['SMA_long'][short_window:], 1, 0)
    df['position'] = df['signal'].diff()
    return df

def simulate_trading(df, initial_balance=1000):
    """Simulates trading based on generated signals and calculates final balance."""
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

    return balance

def plot_results(df):
    """Plots the results of the trading strategy."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='BTC/USDT Price')
    plt.plot(df.index, df['SMA_short'], label='SMA Short', alpha=0.7)
    plt.plot(df.index, df['SMA_long'], label='SMA Long', alpha=0.7)
    plt.scatter(df.index[df['position'] == 1], df['close'][df['position'] == 1], marker='^', color='g', label='Buy Signal', s=100)
    plt.scatter(df.index[df['position'] == -1], df['close'][df['position'] == -1], marker='v', color='r', label='Sell Signal', s=100)
    plt.title('BTC/USDT Trend Following Strategy')
    plt.legend()
    plt.show()

# Main execution
url = create_url()
df = fetch_data(url)
df = preprocess_data(df)
df = calculate_moving_averages(df)
df = generate_signals(df)

initial_balance = 1000
final_balance = simulate_trading(df, initial_balance)
return_on_investment = (final_balance - initial_balance) / initial_balance * 100

print(f'Final Balance: ${final_balance:.2f}')
print(f'Return on Investment: {return_on_investment:.2f}%')

plot_results(df)