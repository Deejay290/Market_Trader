import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import yfinance as yf
from functions import calculations, data_fetcher


class StockPricePredictor:
    """
    Logistic Regression model to predict if stock price will go UP or DOWN.
    Uses technical indicators as features.
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, df):
        """
        Extract technical indicators as features for the model.
        Returns feature matrix and target variable (1 = price goes up, 0 = price goes down).
        """
        df = df.copy()
        
        # Calculate technical indicators
        rsi, macd_line, signal_line = calculations.calculate_technical_indicators(df)
        df['RSI'] = rsi
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        
        # Calculate VWAP
        df = calculations.calculate_vwap(df)
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_10d'] = df['Close'].pct_change(10)
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Price_vs_SMA5'] = (df['Close'] - df['SMA_5']) / df['SMA_5']
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / (df['VWAP'] + 1e-9)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(20).std()
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # MACD features
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Cross'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        # Target variable: Will price go up in next period? (1 = yes, 0 = no)
        # We'll predict if price goes up in the next day
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Select features
        feature_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Cross',
            'Price_Change', 'Price_Change_5d', 'Price_Change_10d',
            'Volume_Change', 'Volume_Ratio',
            'Price_vs_SMA5', 'Price_vs_SMA20', 'Price_vs_VWAP',
            'Volatility', 'High_Low_Range',
            'VWAP'
        ]
        
        # Drop rows with NaN values
        df = df.dropna(subset=feature_cols + ['Target'])
        
        if len(df) == 0:
            return None, None, []
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # Remove last row since target is NaN (no future price)
        X = X[:-1]
        y = y[:-1]
        
        self.feature_names = feature_cols
        return X, y, feature_cols
    
    def train(self, ticker, period='1y', interval='1d'):
        """
        Train the model on historical data for a specific ticker.
        """
        try:
            # Fetch historical data
            df = data_fetcher.get_stock_data(ticker, interval, period)
            if df is None or df.empty:
                return False, "Failed to fetch data"
            
            # Prepare features
            X, y, feature_cols = self.prepare_features(df)
            if X is None or len(X) < 50:
                return False, f"Insufficient data. Need at least 50 samples, got {len(X) if X is not None else 0}"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            return True, {
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(feature_cols)
            }
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def predict(self, ticker):
        """
        Predict if stock price will go UP (1) or DOWN (0) in the next period.
        Returns: (prediction, probability_up, probability_down)
        """
        if not self.is_trained:
            return None, None, None, "Model not trained. Please train first."
        
        try:
            # Get latest data
            df = data_fetcher.get_stock_data(ticker, '1d', '3mo')
            if df is None or df.empty:
                return None, None, None, "Failed to fetch current data"
            
            # Prepare features
            X, _, _ = self.prepare_features(df)
            if X is None or len(X) == 0:
                return None, None, None, "Insufficient data for prediction"
            
            # Use most recent data point
            X_latest = X[-1:].reshape(1, -1)
            X_latest_scaled = self.scaler.transform(X_latest)
            
            # Predict
            prediction = self.model.predict(X_latest_scaled)[0]
            probabilities = self.model.predict_proba(X_latest_scaled)[0]
            
            prob_up = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            prob_down = probabilities[0]
            
            direction = "UP" if prediction == 1 else "DOWN"
            
            return direction, prob_up, prob_down, None
            
        except Exception as e:
            return None, None, None, f"Prediction error: {str(e)}"
    
    def get_feature_importance(self):
        """
        Get feature importance (coefficients) from the logistic regression model.
        """
        if not self.is_trained:
            return None
        
        coefficients = self.model.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return feature_importance


class OptionProfitabilityPredictor:
    """
    Logistic Regression model to predict if an option will be profitable.
    Uses option-specific features.
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features_from_option_data(self, options_df, current_price, dte):
        """
        Prepare features from options chain data.
        """
        if options_df.empty:
            return None, None
        
        # Calculate features for each option
        features_list = []
        targets = []
        
        for idx, row in options_df.iterrows():
            # Option features
            strike = row.get('strike', 0)
            premium = row.get('lastPrice', 0)
            iv = row.get('impliedVolatility', 0)
            volume = row.get('volume', 0)
            oi = row.get('openInterest', 0)
            delta = row.get('delta', 0)
            bid = row.get('bid', 0)
            ask = row.get('ask', 0)
            
            # Moneyness
            moneyness = current_price / strike if strike > 0 else 1.0
            
            # Distance from ATM
            distance_pct = abs(current_price - strike) / current_price * 100
            
            # Spread
            spread = ask - bid if ask > 0 and bid > 0 else 0
            spread_pct = (spread / premium * 100) if premium > 0 else 0
            
            # Liquidity score
            liquidity_score = (volume * 0.5 + oi * 0.5) / 1000  # Normalized
            
            # Time value
            intrinsic = max(0, current_price - strike) if row.get('option_type') == 'call' else max(0, strike - current_price)
            time_value = max(0, premium - intrinsic)
            time_value_pct = (time_value / premium * 100) if premium > 0 else 0
            
            # POP (Probability of Profit)
            pop = calculations.calculate_pop(
                row.get('option_type', 'call'),
                strike, premium, current_price, iv, dte
            )
            pop = pop if not np.isnan(pop) else 0.5
            
            features = [
                moneyness,
                distance_pct,
                iv,
                delta,
                spread_pct,
                liquidity_score,
                time_value_pct,
                pop,
                dte / 365.0,  # Time to expiration in years
                premium / current_price  # Premium as % of stock price
            ]
            
            features_list.append(features)
            
            # Target: Would this option be profitable? (simplified - would need historical data)
            # For now, we'll use POP > 0.5 as proxy
            targets.append(1 if pop > 0.5 else 0)
        
        if not features_list:
            return None, None
        
        X = np.array(features_list)
        y = np.array(targets)
        
        return X, y
    
    def predict_option_profitability(self, option_row, current_price, dte):
        """
        Predict if an option will be profitable.
        """
        if not self.is_trained:
            return None, None, None
        
        try:
            # Convert single row to DataFrame for feature preparation
            options_df = pd.DataFrame([option_row])
            X, _ = self.prepare_features_from_option_data(options_df, current_price, dte)
            
            if X is None or len(X) == 0:
                return None, None, None
            
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            prob_profitable = probabilities[1] if prediction == 1 else probabilities[0]
            prob_unprofitable = probabilities[0] if prediction == 1 else probabilities[1]
            
            return prediction == 1, prob_profitable, prob_unprofitable
            
        except Exception as e:
            return None, None, None

