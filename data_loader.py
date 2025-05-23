import yfinance as yf
import pandas as pd

def load_stock_data(symbol, start_date, end_date):
    """Fetch and preprocess stock data"""
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Add technical indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    
    # Normalize features
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df.dropna()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line