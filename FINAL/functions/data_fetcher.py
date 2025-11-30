import yfinance as yf
import pandas as pd
import streamlit as st
import pytz
from datetime import datetime, timedelta, timezone
import requests

@st.cache_data(ttl=60)
def get_stock_data(ticker_symbol_to_fetch, interval_to_fetch, period_to_fetch):
    """Fetches historical OHLCV data for a given ticker."""
    try:
        df = yf.download(ticker_symbol_to_fetch, interval=interval_to_fetch, period=period_to_fetch, auto_adjust=True, progress=False)
        if not isinstance(df, pd.DataFrame):
             st.error(f"❌ Data fetch for **{ticker_symbol_to_fetch}** returned unexpected type: {type(df)}")
             return None
        if isinstance(df.columns, pd.MultiIndex):
            if ticker_symbol_to_fetch in df.columns.levels[0]:
                 df.columns = df.columns.droplevel(0)
            else:
                 df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.error(f"❌ No data found for **{ticker_symbol_to_fetch}**.")
            return None

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             st.error(f"❌ Missing required columns.")
             return None

        df = df[required_cols].astype(float)
        return df
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None


@st.cache_data(ttl=60)
def get_market_trend_data(ticker_symbol):
    """Fetches historical data across multiple short-term intervals for trend analysis."""
    intervals = ["5m", "15m", "30m", "1h"]
    trend_data = {}
    period_to_fetch = "5d"

    for interval in intervals:
        try:
            data = yf.download(ticker_symbol, interval=interval, period=period_to_fetch, auto_adjust=True, progress=False)
            if isinstance(data, pd.DataFrame) and not data.empty and 'Close' in data.columns:
                trend_data[interval] = data['Close']
            else:
                trend_data[interval] = pd.Series(dtype=float)
        except Exception:
            trend_data[interval] = pd.Series(dtype=float)

    return trend_data


@st.cache_data(ttl=30)
def get_last_price_time(ticker_symbol):
    """Fetches the last recorded price timestamp for a given ticker."""
    try:
        data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        if isinstance(data, pd.DataFrame) and not data.empty:
            last_timestamp = data.index[-1]
            if last_timestamp.tzinfo is None or last_timestamp.tzinfo.utcoffset(last_timestamp) is None:
                eastern = pytz.timezone('US/Eastern')
                last_timestamp = pytz.utc.localize(last_timestamp).tz_convert(eastern)
            else:
                last_timestamp = last_timestamp.tz_convert('US/Eastern')
            return last_timestamp.strftime('%Y-%m-%d %H:%M:%S ET')
        return "N/A"
    except Exception:
        return "N/A"


@st.cache_data(ttl=60)
def get_options_data(ticker_symbol_for_options):
    """Fetches options chain data for a given ticker."""
    try:
        stock = yf.Ticker(ticker_symbol_for_options)
        hist = stock.history(period="5d", interval="1h")
        if hist.empty:
            hist = stock.history(period="1mo", interval="1d")

        if hist.empty:
            info = stock.info
            current_price_options = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price_options is None:
                 return None
        else:
            current_price_options = hist["Close"].iloc[-1]

        expirations = stock.options
        if not expirations:
             return None

        return {
            'current_price': float(current_price_options),
            'expirations': list(expirations),
            'ticker': ticker_symbol_for_options
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_dividend_data(ticker_symbol_div):
    """Fetches dividend information and history for a given ticker."""
    try:
        stock = yf.Ticker(ticker_symbol_div)
        info = stock.info
        dividends = stock.dividends

        if isinstance(dividends, pd.Series) and not dividends.empty:
            dividends.index = pd.to_datetime(dividends.index)
            if not dividends.empty:
                last_div_year = dividends.index.max().year
                dividends = dividends[dividends.index.year >= last_div_year - 10]
        else:
            dividends = pd.Series(dtype=float)

        return info or {}, dividends
    except Exception:
        return {}, pd.Series(dtype=float)


@st.cache_data(ttl=300)
def get_news_from_finnhub(ticker, api_key):
    """Fetches company news from Finnhub API."""
    if not api_key or not isinstance(api_key, str) or len(api_key) < 10:
         return []

    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    news_url = f"https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": from_date, "to": to_date, "token": api_key}

    try:
        news_response = requests.get(news_url, params=params, timeout=10)
        news_response.raise_for_status()
        news_data = news_response.json()
        if isinstance(news_data, list):
            return news_data
        return []
    except Exception:
        return []
