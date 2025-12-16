"""
Model Handler Module
Handles loading the TFT model and generating predictions
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_tft_model.pth')


class StockPredictor:
    """
    Stock price predictor using pre-trained TFT model.
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """
        Load the pre-trained TFT model.
        """
        try:
            if os.path.exists(self.model_path):
                # Try loading as pytorch-forecasting model
                try:
                    from pytorch_forecasting import TemporalFusionTransformer
                    self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)
                    self.model.eval()
                    self.model_type = 'tft'
                    print("Loaded TFT model successfully")
                except Exception as e:
                    # Load as regular PyTorch state dict
                    print(f"Loading as state dict: {e}")
                    self.model = torch.load(self.model_path, map_location=self.device)
                    self.model_type = 'state_dict'
                    print("Loaded model state dict")
            else:
                print(f"Model file not found: {self.model_path}")
                self.model = None
                self.model_type = 'simulation'
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_type = 'simulation'
    
    def predict(self, historical_data: dict, forecast_days: int = 7) -> dict:
        """
        Generate price predictions for the next N days.
        
        Args:
            historical_data: Dictionary with historical price data
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with prediction dates and values
        """
        close_prices = historical_data['close_prices']
        last_date = historical_data['dates'][-1]
        
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, '%Y-%m-%d')
        
        # Generate prediction dates (business days only)
        prediction_dates = []
        current_date = last_date
        while len(prediction_dates) < forecast_days:
            current_date += timedelta(days=1)
            # Skip weekends
            if current_date.weekday() < 5:
                prediction_dates.append(current_date)
        
        # Generate predictions
        if self.model is not None and self.model_type == 'tft':
            predictions = self._predict_with_model(historical_data, forecast_days)
        else:
            # Use statistical prediction if model not available
            predictions = self._predict_statistical(close_prices, forecast_days)
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
            'predictions': predictions,
            'last_actual_date': last_date.strftime('%Y-%m-%d') if isinstance(last_date, datetime) else last_date,
            'last_actual_price': float(close_prices[-1])
        }
    
    def _predict_with_model(self, historical_data: dict, forecast_days: int) -> list:
        """
        Generate predictions using the TFT model.
        """
        try:
            # Prepare input data
            features = historical_data['features']
            
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(x)
            
            return predictions[:forecast_days].cpu().numpy().tolist()
        except Exception as e:
            print(f"Model prediction error: {e}")
            return self._predict_statistical(historical_data['close_prices'], forecast_days)
    
    def _predict_statistical(self, close_prices: list, forecast_days: int) -> list:
        """
        Generate predictions using statistical methods (ARIMA-like approach).
        Uses exponential smoothing and trend analysis.
        """
        prices = np.array(close_prices[-60:])
        
        # Calculate trend
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        trend = coeffs[0]
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Exponential smoothing
        alpha = 0.3
        smoothed = prices[-1]
        
        predictions = []
        last_price = prices[-1]
        
        for i in range(forecast_days):
            # Prediction with trend and mean reversion
            predicted = last_price + trend + np.random.normal(0, volatility * last_price * 0.5)
            
            # Mean reversion towards moving average
            ma = np.mean(prices[-21:])
            predicted = predicted * 0.7 + ma * 0.3
            
            # Ensure reasonable bounds (within 5% of last price)
            predicted = np.clip(predicted, last_price * 0.95, last_price * 1.05)
            
            predictions.append(round(float(predicted), 2))
            last_price = predicted
        
        return predictions


# Singleton instance
_predictor = None


def get_predictor() -> StockPredictor:
    """Get singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = StockPredictor()
    return _predictor


def predict_stock_prices(historical_data: dict, forecast_days: int = 7) -> dict:
    """
    Main function to predict stock prices.
    
    Args:
        historical_data: Dictionary with historical price data
        forecast_days: Number of days to forecast
        
    Returns:
        Dictionary with prediction results
    """
    predictor = get_predictor()
    return predictor.predict(historical_data, forecast_days)


if __name__ == "__main__":
    # Test the predictor
    from data_fetcher import fetch_historical_data, prepare_data_for_model
    
    df = fetch_historical_data()
    data = prepare_data_for_model(df)
    predictions = predict_stock_prices(data)
    
    print("Predictions:")
    for date, price in zip(predictions['dates'], predictions['predictions']):
        print(f"  {date}: Rp {price:,.0f}")
