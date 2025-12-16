"""
Data Fetcher Module
Fetches historical stock data for BBRI.JK from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def fetch_historical_data(symbol: str = "BBRI.JK", days: int = 90) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (default: BBRI.JK)
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    try:
        # Fetch more days to ensure we have enough trading days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Select and rename columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert timezone-aware datetime to timezone-naive
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Take the last N days
        df = df.tail(days).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return sample data if API fails
        return generate_sample_data(days)


def generate_sample_data(days: int = 60) -> pd.DataFrame:
    """
    Generate sample stock data for demonstration purposes.
    """
    np.random.seed(42)
    
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='B')  # Business days
    
    # Generate realistic BBRI stock prices (around 4000-5500 IDR range)
    base_price = 4800
    prices = [base_price]
    
    for i in range(1, days):
        change = np.random.normal(0, 50)  # Random walk
        new_price = prices[-1] + change
        new_price = max(4000, min(5500, new_price))  # Keep in range
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p + np.random.uniform(-20, 20) for p in prices],
        'High': [p + np.random.uniform(10, 50) for p in prices],
        'Low': [p - np.random.uniform(10, 50) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(50000000, 200000000, days)
    })
    
    return df


def prepare_data_for_model(df: pd.DataFrame) -> dict:
    """
    Prepare data in the format required by the TFT model.
    
    Args:
        df: DataFrame with historical stock data
        
    Returns:
        Dictionary with prepared features
    """
    # Use the last 60 days for prediction
    df = df.tail(60).copy()
    
    # Add time-based features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    
    # Calculate technical indicators
    df['returns'] = df['Close'].pct_change()
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['volatility'] = df['returns'].rolling(window=7).std()
    
    # Fill NaN values
    df = df.bfill().ffill()
    
    return {
        'dates': df['Date'].tolist(),
        'close_prices': df['Close'].tolist(),
        'features': df[['Close', 'Volume', 'day_of_week', 'month', 'returns', 'ma7', 'ma21', 'volatility']].values
    }


if __name__ == "__main__":
    # Test the data fetcher
    df = fetch_historical_data()
    print(f"Fetched {len(df)} days of data")
    print(df.tail())
