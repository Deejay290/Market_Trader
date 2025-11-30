import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Import functions from custom modules
from functions import data_fetcher, calculations, options_logic
from display.css_styles import (
    PRIMARY_ACCENT_COLOR_HEX,
    SECONDARY_ACCENT_COLOR_HEX,
    TEXT_LIGHT_COLOR_HEX,
    TEXT_SUBTLE_COLOR_HEX,
    BG_DARK_COLOR_HEX,
    BG_DARKER_COLOR_HEX,
    PANEL_BG_COLOR_CSS,
    BORDER_COLOR_CSS,
    DIVIDER_COLOR_CSS,
    SUCCESS_COLOR_HEX,
    DANGER_COLOR_HEX,
    WARNING_COLOR_HEX,
    INFO_COLOR_HEX
)

# --- Header & Market Status ---

def display_header():
    st.markdown("""
    <div class="header">
        <h1>Market Trader</h1>
        <p>Your AI-powered algorithmic trading dashboard. Analyze real-time market data, identify high-probability options, and track your portfolio.</p>
    </div>
    """, unsafe_allow_html=True)

def display_market_status_panel(market_status_msg, market_status_icon, market_status_color, current_ticker, last_price_time):
    st.markdown(f"""
    <div style="
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        padding: 0.75rem 1.5rem; 
        background-color: {BG_DARKER_COLOR_HEX}; 
        border-radius: 10px; 
        border: 1px solid {BORDER_COLOR_CSS}; 
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.25rem;">{market_status_icon}</span>
            <span style="font-weight: 600; font-size: 1.1rem; color: {TEXT_LIGHT_COLOR_HEX};">{market_status_msg}</span>
        </div>
        <div style="text-align: right;">
            <span style="color: {TEXT_SUBTLE_COLOR_HEX}; font-size: 0.9rem; display: block;">Last Price for {current_ticker}:</span>
            <span style="color: {TEXT_LIGHT_COLOR_HEX}; font-size: 0.9rem; font-weight: 500;">{last_price_time}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Dashboard Metrics & Overview ---

def get_current_metrics(df):
    if df is None or df.empty:
        return 0, 0, 0, 0
    
    current_price = df['Close'].iloc[-1]
    
    if len(df['Close']) > 1:
        prev_close = df['Close'].iloc[-2]
        price_change = current_price - prev_close
        percent_change = (price_change / prev_close) * 100
    else:
        open_price = df['Open'].iloc[-1]
        price_change = current_price - open_price
        percent_change = (price_change / open_price) * 100

    vwap = df['VWAP'].iloc[-1] if 'VWAP' in df.columns else 0
    vwap_diff = ((current_price - vwap) / vwap) * 100 if vwap != 0 else 0
    
    return current_price, price_change, percent_change, vwap_diff

def display_current_metrics(current_price, price_change, percent_change, volume, vwap_diff):
    
    price_color = SUCCESS_COLOR_HEX if price_change >= 0 else DANGER_COLOR_HEX
    change_symbol = "▲" if price_change >= 0 else "▼"
    
    vwap_color = SUCCESS_COLOR_HEX if vwap_diff >= 0 else DANGER_COLOR_HEX
    vwap_text = "Above VWAP" if vwap_diff >= 0 else "Below VWAP"

    st.markdown(f"""
    <div class="section" style="padding: 1.5rem;">
    <div style="
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
        gap: 1.5rem;
    ">
        <div class="metric-box">
            <h4>Current Price</h4>
            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                <h3 style="color: var(--text-light);">${current_price:,.2f}</h3>
                <span class="positive" style="font-size: 1.2rem; font-weight: 600; color: {price_color};">
                    {change_symbol} {percent_change:+.2f}%
                </span>
            </div>
        </div>
        
        <div class="metric-box">
            <h4>Day's Change</h4>
            <h3 style="color: {price_color};">{price_change:+.2f}</h3>
        </div>
        
        <div class="metric-box">
            <h4>Volume</h4>
            <h3>{volume:,.0f}</h3>
        </div>
        
        <div class="metric-box">
            <h4>VWAP Status</h4>
            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                <h3 style="color: {vwap_color};">{vwap_text}</h3>
                <span class="positive" style="font-size: 1.2rem; font-weight: 600; color: {vwap_color};">
                    {vwap_diff:+.2f}%
                </span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        required_keys = ['longName', 'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'longBusinessSummary']
        
        cleaned_info = {key: info.get(key, "N/A") for key in required_keys}
        
        cleaned_info['marketCap'] = info.get('marketCap', 0)
        cleaned_info['trailingPE'] = info.get('trailingPE', 0)
        cleaned_info['forwardPE'] = info.get('forwardPE', 0)
        cleaned_info['fiftyTwoWeekHigh'] = info.get('fiftyTwoWeekHigh', 0)
        cleaned_info['fiftyTwoWeekLow'] = info.get('fiftyTwoWeekLow', 0)

        return cleaned_info
    except Exception:
        return {key: "N/A" for key in required_keys}


def display_company_overview(ticker):
    info = get_company_info(ticker)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"## {info.get('longName', ticker)}")
        st.markdown(f"**{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}**")
        
        summary = info.get('longBusinessSummary', 'No summary available.')
        if summary != 'N/A' and len(summary) > 500:
             summary = summary[:500] + "..."
        st.markdown(f"""
        <p style="color: var(--text-subtle); font-size: 0.95rem; line-height: 1.6; margin-top: 1rem;">
            {summary}
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Key Fundamentals")
        
        def format_large_number(num):
            if num == 0 or num == 'N/A': return "N/A"
            if num > 1e12: return f"${num/1e12:.2f} T"
            if num > 1e9: return f"${num/1e9:.2f} B"
            if num > 1e6: return f"${num/1e6:.2f} M"
            return f"${num:,.2f}"

        st.markdown(f"""
        <div style="
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 1rem; 
            font-size: 0.95rem; 
            margin-top: 1rem;
        ">
            <div>
                <span style="color: var(--text-subtle); display: block;">Market Cap</span>
                <strong style="font-size: 1.1rem;">{format_large_number(info['marketCap'])}</strong>
            </div>
            <div>
                <span style="color: var(--text-subtle); display: block;">P/E (TTM)</span>
                <strong style="font-size: 1.1rem;">{info['trailingPE']:.2f}</strong>
            </div>
            <div>
                <span style="color: var(--text-subtle); display: block;">Fwd P/E</span>
                <strong style="font-size: 1.1rem;">{info['forwardPE']:.2f}</strong>
            </div>
            <div>
                <span style="color: var(--text-subtle); display: block;">52W Range</span>
                <strong style="font-size: 1.1rem;">{info['fiftyTwoWeekLow']:.2f} - {info['fiftyTwoWeekHigh']:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data(ttl=600)
def get_top_stock_metrics(stock_list):
    metrics = []
    for ticker in stock_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if not hist.empty and len(hist) > 1:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change_pct = ((current_price - prev_close) / prev_close) * 100
                metrics.append({
                    'symbol': ticker,
                    'price': f"${current_price:.2f}",
                    'change_pct': change_pct
                })
        except Exception:
            pass 
    return metrics

def display_top_insights(stock_list, api_key, sentiment_analyzer):
    st.markdown("## 📊 Market Insights")
    
    metrics = get_top_stock_metrics(stock_list)
    
    if not metrics:
        st.info("Could not fetch market insight data.")
        return
        
    cols = st.columns(len(metrics) if len(metrics) <= 5 else 5)
    
    for i, metric in enumerate(metrics[:5]):
        with cols[i]:
            change_color = SUCCESS_COLOR_HEX if metric['change_pct'] >= 0 else DANGER_COLOR_HEX
            st.markdown(f"""
            <div class="metric-box" style="text-align: center; padding: 1.2rem;">
                <h4 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-light);">{metric['symbol']}</h4>
                <h3 style="font-size: 1.5rem; margin-bottom: 0.5rem;">{metric['price']}</h3>
                <span style="color: {change_color}; font-weight: 600; font-size: 1rem;">
                    {metric['change_pct']:+.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)


# --- Tab 1: Live Charts ---

def display_chart_analysis(df, rsi, macd, signal, support_levels, resistance_levels, current_price):
    st.markdown("## Price Action & Technicals")
    
    st.markdown("#### Candlestick Chart")
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick',
        increasing_line_color=SUCCESS_COLOR_HEX,
        decreasing_line_color=DANGER_COLOR_HEX
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VWAP'],
        mode='lines',
        name='VWAP',
        line=dict(color=PRIMARY_ACCENT_COLOR_HEX, width=1.5, dash='dash')
    ), row=1, col=1)

    for i, level in enumerate(support_levels[-3:]):
        fig.add_hline(
            y=level, line_width=1, line_dash="dash", 
            line_color=SUCCESS_COLOR_HEX, opacity=0.7,
            annotation_text=f"Support {i+1}", 
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=SUCCESS_COLOR_HEX),
            row=1, col=1
        )

    for i, level in enumerate(resistance_levels[-3:]):
        fig.add_hline(
            y=level, line_width=1, line_dash="dash", 
            line_color=DANGER_COLOR_HEX, opacity=0.7,
            annotation_text=f"Resistance {i+1}", 
            annotation_position="top right",
            annotation_font=dict(size=10, color=DANGER_COLOR_HEX),
            row=1, col=1
        )
        
    fig.add_hline(
        y=current_price, line_width=1, line_dash="dot",
        line_color=WARNING_COLOR_HEX,
        annotation_text="Current Price",
        annotation_position="bottom left",
        annotation_font=dict(size=10, color=WARNING_COLOR_HEX),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(
        x=rsi.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color=INFO_COLOR_HEX, width=1.5)
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color=DANGER_COLOR_HEX, opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color=SUCCESS_COLOR_HEX, opacity=0.5, row=2, col=1)

    fig.update_layout(
        title=f"Price Chart (RSI)",
        height=600,
        plot_bgcolor=PANEL_BG_COLOR_CSS,
        paper_bgcolor=PANEL_BG_COLOR_CSS,
        font_color=TEXT_LIGHT_COLOR_HEX,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_yanchor="top",
        legend_y=1.1,
        legend_xanchor="left",
        legend_x=0,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(gridcolor=DIVIDER_COLOR_CSS, showgrid=True),
        yaxis=dict(gridcolor=DIVIDER_COLOR_CSS, showgrid=True),
        yaxis2=dict(title="RSI", gridcolor=DIVIDER_COLOR_CSS, showgrid=True),
        xaxis_showticklabels=True,
        xaxis2_showticklabels=True
    )
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor=BORDER_COLOR_CSS, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=BORDER_COLOR_CSS, mirror=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Support & Resistance")
        
        st.markdown("##### Resistance")
        if not resistance_levels:
            st.markdown(
                '<div class="sr-level resistance"><span>No significant resistance levels found.</span></div>', 
                unsafe_allow_html=True
            )
        for level in resistance_levels[-3:]:
            st.markdown(
                f'<div class="sr-level resistance"><strong>${level:,.2f}</strong> <span>Potential Sell Zone</span></div>', 
                unsafe_allow_html=True
            )
            
        st.markdown("##### Support")
        if not support_levels:
            st.markdown(
                '<div class="sr-level support"><span>No significant support levels found.</span></div>', 
                unsafe_allow_html=True
            )
        for level in support_levels[-3:]:
            st.markdown(
                f'<div class="sr-level support"><strong>${level:,.2f}</strong> <span>Potential Buy Zone</span></div>', 
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### MACD")
        
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=macd.index, y=macd, mode='lines', name='MACD Line',
            line=dict(color=PRIMARY_ACCENT_COLOR_HEX, width=1.5)
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=signal.index, y=signal, mode='lines', name='Signal Line',
            line=dict(color=WARNING_COLOR_HEX, width=1.5, dash='dash')
        ))
        
        histogram = macd - signal
        colors = [SUCCESS_COLOR_HEX if val >= 0 else DANGER_COLOR_HEX for val in histogram]
        fig_macd.add_trace(go.Bar(
            x=histogram.index, y=histogram, name='Histogram', marker_color=colors
        ))
        
        fig_macd.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            height=300,
            plot_bgcolor=PANEL_BG_COLOR_CSS,
            paper_bgcolor=PANEL_BG_COLOR_CSS,
            font_color=TEXT_LIGHT_COLOR_HEX,
            legend_orientation="h",
            legend_yanchor="top",
            legend_y=1.15,
            legend_xanchor="left",
            legend_x=0,
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(gridcolor=DIVIDER_COLOR_CSS),
            yaxis=dict(gridcolor=DIVIDER_COLOR_CSS, zerolinecolor=BORDER_COLOR_CSS, zerolinewidth=1)
        )
        st.plotly_chart(fig_macd, use_container_width=True)


# --- Tab 2: Market Pulse (News) ---

def display_market_pulse(ticker, api_key, sentiment_analyzer):
    st.markdown("## Market Pulse & News")
    
    news_articles = data_fetcher.get_news_from_finnhub(ticker, api_key)
    
    if not news_articles:
        st.info("No recent news found for this ticker.")
        return

    sentiment_df = calculations.analyze_sentiment_for_articles_vader(news_articles, sentiment_analyzer)
    
    if not sentiment_df.empty:
        avg_score = sentiment_df['sentiment_score'].mean()
        
        if avg_score >= 0.05:
            sentiment_label, sentiment_color, sentiment_icon = "Positive", SUCCESS_COLOR_HEX, "😄"
        elif avg_score <= -0.05:
            sentiment_label, sentiment_color, sentiment_icon = "Negative", DANGER_COLOR_HEX, "😞"
        else:
            sentiment_label, sentiment_color, sentiment_icon = "Neutral", TEXT_SUBTLE_COLOR_HEX, "😐"
        
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom: 2rem; text-align: center; background: {PANEL_BG_COLOR_CSS};">
            <h4 style="color: var(--text-subtle);">Overall News Sentiment (7-Day)</h4>
            <h3 style="color: {sentiment_color}; font-size: 2.5rem; margin-bottom: 0.5rem;">
                {sentiment_icon} {sentiment_label}
            </h3>
            <span style="font-size: 1rem; color: var(--text-light);">Average Score: {avg_score:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

    for article in news_articles[:10]:
        headline = article.get('headline', 'No Headline')
        summary = article.get('summary', 'No Summary')
        source = article.get('source', 'N/A')
        url = article.get('url', '#')
        
        if len(summary) > 150: summary = summary[:150] + "..."
            
        st.markdown(f"""
        <div class="news-card">
            <h4>{headline}</h4>
            <small>Source: {source}</small>
            <p>{summary}</p>
            <a href="{url}" target="_blank">Read Full Article &rarr;</a>
        </div>
        """, unsafe_allow_html=True)


# --- Tab 3: OPTIONS FLOW - COMPLETELY REDESIGNED ---

def analyze_trend_properly(ticker_symbol):
    """
    Properly analyze trend using multiple timeframes with better logic
    """
    try:
        # Fetch data for different timeframes
        data_1d = yf.download(ticker_symbol, period="5d", interval="1d", progress=False, auto_adjust=True)
        data_1h = yf.download(ticker_symbol, period="5d", interval="1h", progress=False, auto_adjust=True)
        data_30m = yf.download(ticker_symbol, period="2d", interval="30m", progress=False, auto_adjust=True)
        data_15m = yf.download(ticker_symbol, period="1d", interval="15m", progress=False, auto_adjust=True)
        
        signals = []
        weights = []
        details = []
        
        # 1-Day Analysis (Weight: 40%)
        if not data_1d.empty and len(data_1d) >= 3:
            # Extract Close column properly
            if isinstance(data_1d.columns, pd.MultiIndex):
                close_col = data_1d['Close'].iloc[:, 0] if isinstance(data_1d['Close'], pd.DataFrame) else data_1d['Close']
            else:
                close_col = data_1d['Close']
            
            close_prices = close_col.values
            
            # Ensure we have scalar values
            if len(close_prices) >= 3:
                sma_3 = float(np.mean(close_prices[-3:]))
                current = float(close_prices[-1])
                prev = float(close_prices[-2])
                
                day_change = ((current - prev) / prev) * 100
                
                if current > sma_3 and day_change > 0.5:
                    signals.append(1)
                    details.append(f"📈 Daily: UP {day_change:+.2f}% (Above 3-day avg)")
                elif current < sma_3 and day_change < -0.5:
                    signals.append(-1)
                    details.append(f"📉 Daily: DOWN {day_change:+.2f}% (Below 3-day avg)")
                else:
                    signals.append(0)
                    details.append(f"➖ Daily: FLAT {day_change:+.2f}%")
                weights.append(0.40)
        
        # 1-Hour Analysis (Weight: 30%)
        if not data_1h.empty and len(data_1h) >= 6:
            if isinstance(data_1h.columns, pd.MultiIndex):
                close_col = data_1h['Close'].iloc[:, 0] if isinstance(data_1h['Close'], pd.DataFrame) else data_1h['Close']
            else:
                close_col = data_1h['Close']
            
            close_prices = close_col.values
            
            if len(close_prices) >= 6:
                sma_6 = float(np.mean(close_prices[-6:]))
                current = float(close_prices[-1])
                prev_6 = float(close_prices[-6])
                change_6h = ((current - prev_6) / prev_6) * 100
                
                if current > sma_6 and change_6h > 0.3:
                    signals.append(1)
                    details.append(f"📈 Hourly: UP {change_6h:+.2f}% (6h trend)")
                elif current < sma_6 and change_6h < -0.3:
                    signals.append(-1)
                    details.append(f"📉 Hourly: DOWN {change_6h:+.2f}% (6h trend)")
                else:
                    signals.append(0)
                    details.append(f"➖ Hourly: FLAT {change_6h:+.2f}%")
                weights.append(0.30)
        
        # 30-Minute Analysis (Weight: 20%)
        if not data_30m.empty and len(data_30m) >= 4:
            if isinstance(data_30m.columns, pd.MultiIndex):
                close_col = data_30m['Close'].iloc[:, 0] if isinstance(data_30m['Close'], pd.DataFrame) else data_30m['Close']
            else:
                close_col = data_30m['Close']
            
            close_prices = close_col.values
            
            if len(close_prices) >= 4:
                current = float(close_prices[-1])
                prev_4 = float(close_prices[-4])
                change_2h = ((current - prev_4) / prev_4) * 100
                
                if change_2h > 0.2:
                    signals.append(1)
                    details.append(f"📈 30min: UP {change_2h:+.2f}% (2h momentum)")
                elif change_2h < -0.2:
                    signals.append(-1)
                    details.append(f"📉 30min: DOWN {change_2h:+.2f}% (2h momentum)")
                else:
                    signals.append(0)
                    details.append(f"➖ 30min: FLAT {change_2h:+.2f}%")
                weights.append(0.20)
        
        # 15-Minute Analysis (Weight: 10%)
        if not data_15m.empty and len(data_15m) >= 4:
            if isinstance(data_15m.columns, pd.MultiIndex):
                close_col = data_15m['Close'].iloc[:, 0] if isinstance(data_15m['Close'], pd.DataFrame) else data_15m['Close']
            else:
                close_col = data_15m['Close']
            
            close_prices = close_col.values
            
            if len(close_prices) >= 4:
                current = float(close_prices[-1])
                prev_4 = float(close_prices[-4])
                change_1h = ((current - prev_4) / prev_4) * 100
                
                if change_1h > 0.15:
                    signals.append(1)
                    details.append(f"📈 15min: UP {change_1h:+.2f}% (1h momentum)")
                elif change_1h < -0.15:
                    signals.append(-1)
                    details.append(f"📉 15min: DOWN {change_1h:+.2f}% (1h momentum)")
                else:
                    signals.append(0)
                    details.append(f"➖ 15min: FLAT {change_1h:+.2f}%")
                weights.append(0.10)
        
        # Calculate weighted score
        if signals and weights:
            weighted_score = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
            confidence = abs(weighted_score) * 100
            
            if weighted_score > 0.15:
                trend = "Bullish"
            elif weighted_score < -0.15:
                trend = "Bearish"
            else:
                trend = "Neutral"
            
            return trend, confidence, details
        else:
            return "Neutral", 0, ["⚠️ Insufficient data for analysis"]
            
    except Exception as e:
        return "Neutral", 0, [f"⚠️ Error: Unable to analyze trend data"]


