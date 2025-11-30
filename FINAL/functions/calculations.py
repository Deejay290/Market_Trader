import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytz
from datetime import datetime, timedelta, timezone
from scipy.stats import norm
import yfinance as yf
import requests

# Define color constants that might be used
SUCCESS_COLOR_HEX = "#10b981"
TEXT_SUBTLE_COLOR_HEX = "#94a3b8"

def get_market_status():
    """Clean market status."""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    is_weekday = now.weekday() < 5
    is_market_hours = market_open <= now < market_close

    if is_weekday and is_market_hours:
        return "Market Open", "🟢", SUCCESS_COLOR_HEX, now.strftime('%Y-%m-%d %H:%M ET')
    else:
        return "Market Closed", "⚫", TEXT_SUBTLE_COLOR_HEX, now.strftime('%Y-%m-%d %H:%M ET')

def analyze_market_trend(trend_data):
    """
    Analyzes multi-interval data to determine the current market trend (bullish/bearish).
    Returns a trend direction ('Bullish' or 'Bearish'), a confidence score, and the reasons.
    """
    if not trend_data:
        return "Neutral", 0, ["Not enough data for trend analysis."]

    trend_score = 0
    reasons = []

    weights = {"5m": 0.4, "15m": 0.3, "30m": 0.2, "1h": 0.1}
    valid_intervals = [i for i in trend_data.keys() if i in weights]

    for interval in valid_intervals:
        close_prices = trend_data[interval]
        # Ensure we have at least two data points to compare
        if not isinstance(close_prices, pd.Series) or len(close_prices) < 2:
            reasons.append(f"⚠️ Insufficient data for {interval}")
            continue

        # Explicitly get the first and last numerical price values
        first_price = close_prices.iloc[0]
        last_price = close_prices.iloc[-1]

        # Check if prices are valid numbers
        if not isinstance(first_price, (int, float, np.number)) or not isinstance(last_price, (int, float, np.number)):
             reasons.append(f"⚠️ Non-numeric data for {interval}")
             continue

        # Calculate percentage change, handling potential division by zero
        if first_price == 0:
            price_change_pct = 0.0
        else:
            price_change_pct = (last_price - first_price) / first_price * 100

        # The comparison causing the error - price_change_pct MUST be a single number here
        interval_score = 0
        try:
            if price_change_pct > 0.1:  # Threshold to ignore minor noise
                interval_score = 1
                reasons.append(f"📈 {interval} is UP {price_change_pct:+.2f}%")
            elif price_change_pct < -0.1:
                interval_score = -1
                reasons.append(f"📉 {interval} is DOWN {price_change_pct:+.2f}%")
            else:
                reasons.append(f"➖ {interval} is Flat ({price_change_pct:+.2f}%)")
        except Exception as e:
            # Catch unexpected comparison errors
            reasons.append(f"⚠️ Error comparing {interval}: {e}")
            continue # Skip this interval if comparison fails


        trend_score += interval_score * weights[interval]

    # Determine final trend based on the weighted score
    if trend_score > 0.1:
        trend_direction = "Bullish"
    elif trend_score < -0.1:
        trend_direction = "Bearish"
    else:
        trend_direction = "Neutral"

    # Max possible score using weights sum (0.4 + 0.3 + 0.2 + 0.1 = 1.0)
    max_possible_score = sum(weights[i] for i in valid_intervals if len(trend_data.get(i, [])) >= 2)
    if max_possible_score == 0:
         confidence = 0 # Avoid division by zero if no valid intervals
    else:
         confidence = min(abs(trend_score) / max_possible_score * 100, 100)

    return trend_direction, confidence, reasons


# List of top stocks for overview
OVERVIEW_STOCKS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "JPM", "SMCI", "GOOG", "TSM", "ASML", "CRM"]

@st.cache_resource
def load_sentiment_analyzer_global():
    with st.spinner("Loading global sentiment analyzer..."):
        try:
            # Check if lexicon is available without downloading first
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            try:
                nltk.download('vader_lexicon', quiet=True)
            except Exception as e:
                st.error(f"Failed to download vader_lexicon: {e}")
                return None # Return None if download fails
        # Initialize only if lexicon is found or downloaded
        try:
            return SentimentIntensityAnalyzer()
        except Exception as e:
            st.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
            return None


def analyze_sentiment_for_articles_vader(articles, analyzer):
    # Check if analyzer was loaded successfully
    if analyzer is None:
        return pd.DataFrame() # Return empty DataFrame if analyzer failed to load

    sentiment_data = []
    if not articles: return pd.DataFrame()

    for article in articles:
        text_to_analyze = f"{article.get('headline', '')}. {article.get('summary', '')}"
        if text_to_analyze.strip():
            try:
                vs = analyzer.polarity_scores(text_to_analyze)
                compound_score = vs['compound']

                if compound_score >= 0.05: sentiment_label = "POSITIVE"
                elif compound_score <= -0.05: sentiment_label = "NEGATIVE"
                else: sentiment_label = "NEUTRAL"

                sentiment_data.append({
                    'date': datetime.fromtimestamp(article.get('datetime', 0), tz=timezone.utc).date(),
                    'sentiment_score': compound_score,
                    'sentiment_label': sentiment_label,
                    'headline': article.get('headline', '')
                })
            except Exception: pass # Ignore errors for single articles
    return pd.DataFrame(sentiment_data)

def calculate_vwap(df):
    df = df.copy()
    if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        # Ensure Volume column is numeric and handle potential NaNs
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['price_volume_weighted'] = df['TypicalPrice'] * df['Volume']
        df['CumVol'] = df['Volume'].cumsum()
        df['CumVolPrice'] = df['price_volume_weighted'].cumsum()
        # Add small epsilon to prevent division by zero if CumVol is 0
        df['VWAP'] = df['CumVolPrice'] / (df['CumVol'].replace(0, np.nan) + 1e-10)
        df['VWAP'] = df['VWAP'].ffill().bfill() # Forward fill then backward fill NaNs
        df.drop(columns=['TypicalPrice', 'price_volume_weighted'], inplace=True, errors='ignore')
    else:
        df['VWAP'] = np.nan # Assign NaN if required columns are missing
    return df


def find_pivots(series, window=5):
    if not isinstance(series, pd.Series) or series.empty:
        return [], []

    if window % 2 == 0: window += 1
    if window < 3: window = 3

    # Ensure window size is not larger than the series length
    if len(series) < window:
        return [], [] # Cannot calculate pivots if series is too short

    min_val = series.rolling(window=window, center=True, min_periods=1).min() # Use min_periods=1 for edges
    max_val = series.rolling(window=window, center=True, min_periods=1).max()

    # Drop NaNs that might result from rolling operation edge cases
    supports_raw = series[series == min_val].dropna()
    resistances_raw = series[series == max_val].dropna()

    tolerance_factor = 0.005
    series_mean_fallback = series.mean() # Pre-calculate mean as fallback

    unique_supports = []
    if not supports_raw.empty:
        sorted_supports = sorted(supports_raw.unique())
        if sorted_supports:
            unique_supports.append(sorted_supports[0])
            for val in sorted_supports[1:]:
                ref_price = unique_supports[-1] if unique_supports[-1] != 0 else series_mean_fallback
                threshold = ref_price * tolerance_factor if pd.notna(ref_price) else 0.01 # Handle potential NaN mean
                if pd.notna(threshold) and abs(val - unique_supports[-1]) > threshold:
                    unique_supports.append(val)
                elif len(unique_supports) > 0: # Ensure list is not empty before averaging
                    unique_supports[-1] = (unique_supports[-1] + val) / 2

    unique_resistances = []
    if not resistances_raw.empty:
        sorted_resistances = sorted(resistances_raw.unique(), reverse=True)
        if sorted_resistances:
            unique_resistances.append(sorted_resistances[0])
            for val in sorted_resistances[1:]:
                ref_price = unique_resistances[-1] if unique_resistances[-1] != 0 else series_mean_fallback
                threshold = ref_price * tolerance_factor if pd.notna(ref_price) else 0.01
                if pd.notna(threshold) and abs(val - unique_resistances[-1]) > threshold:
                    unique_resistances.append(val)
                elif len(unique_resistances) > 0:
                     unique_resistances[-1] = (unique_resistances[-1] + val) / 2

    return sorted(unique_supports), sorted(unique_resistances, reverse=True)


def calculate_technical_indicators(df):
    rsi, macd_line, signal_line = pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    if 'Close' not in df.columns or df['Close'].empty or df['Close'].isna().all():
        return rsi, macd_line, signal_line # Return empty Series if no valid Close data

    close_prices = df['Close'].dropna() # Drop NaNs before calculations
    if close_prices.empty:
         return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # RSI
    RSI_PERIOD = 14
    if len(close_prices) <= RSI_PERIOD: # Need enough data for RSI
        rsi = pd.Series([50] * len(df.index), index=df.index) # Default to 50 if not enough data
    else:
        delta = close_prices.diff()
        gain = delta.clip(lower=0).fillna(0) # Fill NaNs created by diff
        loss = -delta.clip(upper=0).fillna(0)

        # Use simple moving average for the first calculation then switch to exponential
        avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()[:RSI_PERIOD]
        avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()[:RSI_PERIOD]

        # Apply EWM for subsequent periods
        avg_gain = pd.concat([avg_gain, gain[RSI_PERIOD:]]).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
        avg_loss = pd.concat([avg_loss, loss[RSI_PERIOD:]]).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()


        rs = avg_gain / (avg_loss.replace(0, 1e-10))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.reindex(df.index).fillna(50) # Reindex to original DataFrame index and fill NaNs

    # MACD
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    if len(close_prices) >= MACD_SLOW_PERIOD: # Need enough data for MACD
        ema_fast = close_prices.ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
        ema_slow = close_prices.ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).reindex(df.index) # Reindex
        signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean().reindex(df.index) # Reindex
        # Fill initial NaNs that result from EMA calculations
        macd_line = macd_line.fillna(0)
        signal_line = signal_line.fillna(0)
    else:
        # If not enough data, return empty series aligned with the main df index
        macd_line = pd.Series([0.0] * len(df.index), index=df.index)
        signal_line = pd.Series([0.0] * len(df.index), index=df.index)


    return rsi, macd_line, signal_line

def calculate_pop(option_type, strike_price, premium, current_price, implied_volatility, dte):
    if not all(isinstance(x, (int, float, np.number)) for x in [strike_price, premium, current_price, implied_volatility, dte]):
        return np.nan # Ensure all inputs are numeric
    if dte <= 0 or implied_volatility <= 0 or current_price <=0 or strike_price <= 0:
        return np.nan # Basic validity checks

    time_to_expiration_years = dte / 365.0
    r = 0.05 # Assume 5% risk-free rate
    q = 0.0 # Assume no dividend yield

    sigma_sqrt_t = implied_volatility * np.sqrt(time_to_expiration_years)
    if sigma_sqrt_t == 0: return np.nan

    try:
        # Use log(current_price / strike_price) for d1 calculation
        d1 = (np.log(current_price / strike_price) + (r - q + 0.5 * implied_volatility**2) * time_to_expiration_years) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        if option_type == 'call':
            # Probability of finishing ITM for a call is N(d2)
            pop = norm.cdf(d2)
        elif option_type == 'put':
            # Probability of finishing ITM for a put is N(-d2)
            pop = norm.cdf(-d2)
        else:
            pop = np.nan

    except (ZeroDivisionError, ValueError, OverflowError):
        return np.nan

    # Ensure POP is between 0 and 1
    return np.clip(pop, 0, 1) if pd.notna(pop) else np.nan