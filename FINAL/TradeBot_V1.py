import streamlit as st
import yfinance as yf
import pandas as pd
from io import BytesIO
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm # Import norm for POP calculation

# --- Config ---
# Reading API_KEY for Finnhub from secrets.toml
API_KEY = st.secrets["API_KEY"]
st.set_page_config(
    layout="wide",
    page_title="Market Trader",
   
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Define Colors in Python (for both CSS and Plotly/Matplotlib) ---
# Use HEX codes for all colors that will be used by Plotly/Matplotlib
PRIMARY_ACCENT_COLOR_HEX = "#00C6FF"
SECONDARY_ACCENT_COLOR_HEX = "#0072FF"
TEXT_LIGHT_COLOR_HEX = "#E0E6EB"
TEXT_SUBTLE_COLOR_HEX = "#8A99AC"
BG_DARK_COLOR_HEX = "#1A1D24"
BG_DARKER_COLOR_HEX = "#121417"

# Define solid HEX colors for Plotly/Matplotlib borders/grids.
# These are explicitly hex codes for Matplotlib/Plotly's strict parsing.
BORDER_COLOR_FOR_MPL_PLOTLY = "#3C414B" # A solid grey for borders/grids
DIVIDER_COLOR_FOR_MPL_PLOTLY = "#2C2F36" # A darker solid grey for dividers/grids

# For CSS, you can still define and use RGBA for transparency where desired
PANEL_BG_COLOR_CSS = "rgba(30, 33, 40, 0.9)"
BORDER_COLOR_CSS = "rgba(60, 65, 75, 0.6)"
DIVIDER_COLOR_CSS = "rgba(255, 255, 255, 0.08)"

SUCCESS_COLOR_HEX = "#00C853"
DANGER_COLOR_HEX = "#FF5252"
WARNING_COLOR_HEX = "#FFD600"
INFO_COLOR_HEX = "#2196F3"


# --- Inject Custom CSS ---
st.markdown(f"""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables (Adjusted for sleek, modern feel) */
    :root {{
        --primary-accent: {PRIMARY_ACCENT_COLOR_HEX};
        --secondary-accent: {SECONDARY_ACCENT_COLOR_HEX};
        --text-light: {TEXT_LIGHT_COLOR_HEX};
        --text-subtle: {TEXT_SUBTLE_COLOR_HEX};
        --bg-dark: {BG_DARK_COLOR_HEX};
        --bg-darker: {BG_DARKER_COLOR_HEX};
        --panel-bg: {PANEL_BG_COLOR_CSS}; /* Use RGBA for CSS transparency */
        --border-color: {BORDER_COLOR_CSS}; /* Use RGBA for CSS transparency */
        --divider-color: {DIVIDER_COLOR_CSS}; /* Use RGBA for CSS transparency */
        --success-color: {SUCCESS_COLOR_HEX};
        --danger-color: {DANGER_COLOR_HEX};
        --warning-color: {WARNING_COLOR_HEX};
        --info-color: {INFO_COLOR_HEX};
    }}
    
    body {{
        background-color: var(--bg-dark);
        color: var(--text-light);
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        overflow-x: hidden; /* Prevent horizontal scroll */
    }}

    /* General Streamlit Overrides */
    .stApp {{
        background-color: var(--bg-dark);
        color: var(--text-light); /* Ensure app text uses light color */
    }}
    
    .main .block-container {{
        padding: 1.5rem 3rem;
        max-width: 1600px; /* Slightly wider for more content */
        margin-left: auto;
        margin-right: auto;
    }}
    
    /* Header Section */
    .header {{
        background: linear-gradient(135deg, var(--bg-darker) 0%, rgba(28, 31, 38, 0.9) 100%);
        padding: 2rem 3rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        border: 1px solid var(--border-color);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }}
    
    .header h1 {{
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
        background: linear-gradient(to right, #FFFFFF 20%, #AEC6FF 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.06em;
    }}
    
    .header p {{
        opacity: 0.95;
        font-size: 1.15rem;
        line-height: 1.6;
        color: var(--text-subtle);
    }}

    /* Section Containers */
    .section {{
        background: var(--panel-bg);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        border: 1px solid var(--border-color);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }}
    
    .section h2 {{
        color: var(--text-light);
        font-size: 2.0rem;
        font-weight: 700;
        border-bottom: 1px solid var(--divider-color);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }}

    /* Metric Boxes */
    .metric-box {{
        background: var(--panel-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease-in-out;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
        border-color: var(--primary-accent);
    }}

    /* Subtle border glow on hover for metric boxes */
    .metric-box::before {{
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--primary-accent), var(--secondary-accent));
        z-index: -1;
        opacity: 0;
        filter: blur(8px);
        transition: opacity 0.3s ease-in-out;
        border-radius: 12px;
    }}

    .metric-box:hover::before {{
        opacity: 0.6;
    }}
    
    .metric-box h3 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: var(--text-light);
    }}
    
    .metric-box h4 {{
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-subtle);
    }}
    
    .positive {{ color: var(--success-color); }}
    .negative {{ color: var(--danger-color); }}
    .neutral {{ color: var(--warning-color); }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 16px;
        margin-bottom: 1.5rem;
        justify-content: center;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: var(--bg-darker);
        border-radius: 8px !important;
        padding: 0.9rem 1.8rem !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-subtle) !important;
        transition: all 0.2s ease;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(40, 45, 55, 0.8) !important;
        color: var(--text-light) !important;
        transform: translateY(-2px);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent)) !important;
        color: white !important;
        font-weight: 700;
        border-color: var(--primary-accent) !important;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3), 0 0 20px rgba(0, 114, 255, 0.2);
        transform: translateY(-2px);
    }}

    /* Form Elements (Selectbox, TextInput) */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput input,
    .stTextInput textarea {{
        background-color: var(--bg-darker) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-light) !important;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        height: auto;
        font-size: 1rem;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .stSelectbox div[data-baseweb="select"] {{
        border-radius: 8px;
    }}
    
    /* Reinforce the dropdown (currently selected value) text color */
    .stSelectbox div[data-baseweb="select"] > div span {{
        color: var(--text-light) !important;
    }}
    
    .stSelectbox div[data-baseweb="select"] > div:focus-visible,
    .stTextInput input:focus-visible,
    .stTextInput textarea:focus-visible {{
        border-color: var(--primary-accent) !important;
        box-shadow: 0 0 0 3px rgba(0, 198, 255, 0.4) !important;
        outline: none;
    }}

    /* FIX: Dropdown menu (popover) styling */
    div[data-baseweb="popover"] div[role="listbox"] {{
        background-color: var(--bg-darker) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        padding: 0.5rem 0;
        max-height: 300px;
        overflow-y: auto;
    }}

    /* FIX: Styling for individual options in the dropdown */
    div[data-baseweb="popover"] [role="option"] {{
        color: var(--text-light) !important;
        background-color: transparent !important;
        transition: background-color 0.2s ease, color 0.2s ease;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem;
        line-height: 1.4;
    }}
    
    div[data-baseweb="popover"] [role="option"] > div > div > div > span,
    div[data-baseweb="popover"] [role="option"] > div > div > span,
    div[data-baseweb="popover"] [role="option"] > span {{
        color: var(--text-light) !important;
    }}

    /* FIX: Hover state for dropdown options */
    div[data-baseweb="popover"] [role="option"]:hover {{
        background-color: rgba(0, 198, 255, 0.15) !important;
        color: var(--text-light) !important;
    }}

    /* FIX: Selected state for dropdown options */
    div[data-baseweb="popover"] [aria-selected="true"] {{
        background-color: var(--primary-accent) !important;
        color: white !important;
        font-weight: 600 !important;
    }}

    /* News Cards */
    .news-card {{
        background: var(--panel-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }}
    
    .news-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        border-color: var(--primary-accent);
    }}
    
    .news-card h4 {{
        color: var(--text-light);
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }}
    
    .news-card small {{
        color: var(--text-subtle);
        font-size: 0.9rem;
    }}
    
    .news-card p {{
        color: var(--text-light);
        opacity: 0.9;
        line-height: 1.6;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }}
    
    .news-card a {{
        color: var(--primary-accent);
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }}
    .news-card a:hover {{
        color: var(--secondary-accent);
        text-decoration: underline;
    }}

    /* Support/Resistance Levels */
    .sr-level {{
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        background: rgba(36, 40, 48, 0.7);
        border-left: 6px solid;
        color: var(--text-light);
        font-size: 1.05rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }}
    
    .sr-level strong {{
        font-weight: 700;
        color: var(--text-light);
    }}

    .sr-level span {{
        color: var(--text-subtle);
        font-size: 0.9rem;
    }}
    
    .support {{ border-left-color: var(--success-color); }}
    .resistance {{ border-left-color: var(--danger-color); }}

    /* Progress Bar */
    progress {{
        height: 10px;
        border-radius: 5px;
        width: 100%;
        margin-top: 0.75rem;
        background-color: var(--border-color);
        border: none;
    }}
    
    progress::-webkit-progress-bar {{
        background-color: var(--border-color);
        border-radius: 5px;
    }}
    
    progress::-webkit-progress-value {{
        background: linear-gradient(to right, var(--primary-accent), var(--secondary-accent));
        border-radius: 5px;
    }}
    
    progress::-moz-progress-bar {{
        background: linear-gradient(to right, var(--primary-accent), var(--secondary-accent));
        border-radius: 5px;
    }}

    /* DataFrame */
    .stDataFrame {{
        border-radius: 10px;
        background-color: var(--panel-bg) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }}
    .stDataFrame table {{
        background-color: transparent !important;
        border-collapse: collapse;
    }}
    .stDataFrame thead th {{
        background-color: rgba(40, 45, 55, 0.9) !important;
        color: var(--text-subtle) !important;
        font-weight: 600 !important;
        border-bottom: 1px solid var(--divider-color) !important;
        padding: 0.8rem 1rem !important;
        text-transform: uppercase;
        font-size: 0.9rem;
    }}
    .stDataFrame tbody tr {{
        background-color: var(--panel-bg) !important;
    }}
    .stDataFrame tbody td {{
        color: var(--text-light) !important;
        border-bottom: 1px solid var(--divider-color) !important;
        padding: 0.7rem 1rem !important;
    }}
    /* Hover for rows */
    .stDataFrame tbody tr:hover {{
        background-color: rgba(40, 45, 55, 0.9) !important;
    }}

    /* Sidebar */
    .stSidebar > div:first-child {{
        background: var(--bg-darker);
        border-right: 1px solid var(--border-color);
        padding: 1.5rem 1.2rem;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
    }}
    .st-emotion-cache-1weq7rx {{
        background-color: var(--bg-darker);
    }}
    .sidebar h3 {{
        color: var(--text-light) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--divider-color);
        margin-bottom: 1.5rem;
    }}

    /* Glow effects - refined */
    .glow-box {{
        box-shadow: 0 0 20px rgba(0, 198, 255, 0.2), 0 0 30px rgba(0, 114, 255, 0.15);
    }}
    
    .glow-text {{
        text-shadow: 0 0 10px rgba(0, 198, 255, 0.5), 0 0 15px rgba(0, 114, 255, 0.3);
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--primary-accent);
        border-radius: 5px;
    }}
    
    /* Custom checkbox - ensure visibility */
    .stCheckbox span {{
        color: var(--text-light) !important;
    }}
    .stCheckbox [data-baseweb="checkbox"] label span:last-child {{
        color: var(--text-light) !important;
    }}

    /* Alert boxes - pulse effect retained but color adjusted */
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(0, 198, 255, 0.7); }}
        70% {{ box-shadow: 0 0 0 10px rgba(0, 198, 255, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 198, 255, 0); }}
    }}
    
    .stAlert {{
        border: 1px solid var(--primary-accent) !important;
        background-color: rgba(0, 198, 255, 0.1) !important;
        color: var(--text-light) !important;
        border-radius: 8px;
        padding: 1rem;
    }}
    .stAlert.st-emotion-cache-p5m9t8.e1f1d6gn0 {{
        animation: pulse 2s infinite;
    }}

    /* General text enhancements */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-light);
    }}
    p, li, div {{
        color: var(--text-light);
    }}
    small {{
        color: var(--text-subtle);
    }}

    /* Streamlit button specific styling */
    .stButton > button {{
        background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 1.6rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
        letter-spacing: 0.03em;
    }}
    .stButton > button:hover {{
        background: linear-gradient(90deg, var(--secondary-accent), var(--primary-accent));
        box-shadow: 0 6px 20px rgba(0, 198, 255, 0.4);
        transform: translateY(-2px);
    }}
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 198, 255, 0.2);
    }}

    /* Help icon styling */
    span[data-testid="stHelpTooltip"] svg {{
        fill: var(--text-subtle) !important;
        opacity: 0.8 !important;
        transition: fill 0.2s ease, opacity 0.2s ease;
    }}
    span[data-testid="stHelpTooltip"] svg:hover {{
        fill: var(--primary-accent) !important;
        opacity: 1 !important;
    }}
    
    /* For the actual help tooltip popover box */
    div[data-baseweb="tooltip"] {{
        background-color: var(--bg-darker) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-light) !important;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }}
    
    div[data-baseweb="tooltip"] .stMarkdown p,
    div[data-baseweb="tooltip"] .stMarkdown {{
        color: var(--text-light) !important;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem 1rem;
        }}
        .header {{
            padding: 1.5rem 1.5rem;
        }}
        .header h1 {{
            font-size: 2rem;
        }}
        .header p {{
            font-size: 1rem;
        }}
        .section {{
            padding: 1.5rem;
        }}
        .metric-box {{
            padding: 1rem;
        }}
        .metric-box h3 {{
            font-size: 2rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            flex-wrap: wrap;
            justify-content: flex-start;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.7rem 1rem !important;
            font-size: 0.9rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header glow-box">
    <h1 class="glow-text">Market Trader</h1>
    <p>Next-gen algorithmic trading platform with AI-powered market intelligence for precision trading.</p>
</div>
""", unsafe_allow_html=True)

# --- Market Status Display (New Section) ---
# Function to check market status
@st.cache_data(ttl=60)
def get_market_status():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

    is_weekday = now.weekday() < 5
    is_market_hours = market_open_time <= now < market_close_time

    status_message = ""
    status_icon = ""
    status_color = ""

    if is_weekday and is_market_hours:
        status_message = "Market is currently OPEN"
        status_icon = "🟢"
        status_color = "green"
    elif not is_weekday:
        status_message = "Market is CLOSED (Weekend)"
        status_icon = "⚪"
        status_color = "gray"
    elif now < market_open_time:
        status_message = "Market is CLOSED (Pre-market)"
        status_icon = "⚪"
        status_color = "gray"
    elif now >= market_close_time:
        status_message = "Market is CLOSED (After-hours)"
        status_icon = "⚪"
        status_color = "gray"
    else:
        status_message = "Market status unknown"
        status_icon = "❓"
        status_color = "yellow"

    return status_message, status_icon, status_color, now.strftime('%Y-%m-%d %H:%M:%S ET')

market_status_msg, market_status_icon, market_status_color, current_et_time = get_market_status()

# Try to get the last recorded price time for a default ticker (e.g., NVDA)
@st.cache_data(ttl=30)
def get_last_price_time(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        if not data.empty:
            last_timestamp = data.index[-1]
            if last_timestamp.tz is None:
                eastern = pytz.timezone('US/Eastern')
                last_timestamp = last_timestamp.tz_localize('UTC').tz_convert(eastern)
            else:
                last_timestamp = last_timestamp.tz_convert('US/Eastern')
            return last_timestamp.strftime('%Y-%m-%d %H:%M:%S ET')
        return "N/A (No data)"
    except Exception:
        return "N/A (Error fetching)"

# Initialize all session state variables at the very top of the script
if 'selected_top_stock_key_widget' not in st.session_state: 
    st.session_state.selected_top_stock_key_widget = ""
if 'manual_ticker_input_value_key' not in st.session_state:
    st.session_state.manual_ticker_input_value_key = "NVDA"
if 'selected_interval_key_widget' not in st.session_state:
    st.session_state.selected_interval_key_widget = "1d"
if 'selected_period_key_widget' not in st.session_state:
    st.session_state.selected_period_key_widget = "3mo"
# Default indicator periods
if 'rsi_period' not in st.session_state:
    st.session_state.rsi_period = 14
if 'macd_fast_period' not in st.session_state:
    st.session_state.macd_fast_period = 12
if 'macd_slow_period' not in st.session_state:
    st.session_state.macd_slow_period = 26
if 'macd_signal_period' not in st.session_state:
    st.session_state.macd_signal_period = 9

# Initialize watchlist in session state, if not already present
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["NVDA", "AAPL", "MSFT"] # Default watchlist items

current_main_ticker = st.session_state.manual_ticker_input_value_key
last_price_recorded_time = get_last_price_time(current_main_ticker)

st.markdown(f"""
<div class="section" style="padding:1rem 2rem; margin-bottom:1.5rem; display:flex; align-items:center;">
    <h3 style="margin:0; flex-grow:1; color:{TEXT_LIGHT_COLOR_HEX}; font-size:1.5rem;">
        {market_status_icon} Market Status: <span style="color:{market_status_color};">{market_status_msg}</span>
    </h3>
    <p style="margin:0; font-size:0.9rem; color:var(--text-subtle);">
        Last Price Data Recorded ({current_main_ticker}): {last_price_recorded_time}
    </p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem; padding-top:1rem;">
        <h3 style="color:var(--text-light); border-bottom:1px solid var(--divider-color); padding-bottom:1rem;">
            🔍 Market Scanner
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    TOP_STOCKS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "JPM"]
    
    def update_ticker_from_dropdown_callback():
        if st.session_state.selected_top_stock_key_widget:
            st.session_state.manual_ticker_input_value_key = st.session_state.selected_top_stock_key_widget

    selected_top_stock = st.selectbox(
        "Select Top Stock", 
        [""] + TOP_STOCKS, 
        key="selected_top_stock_key_widget", 
        on_change=update_ticker_from_dropdown_callback, 
        help="Choose from a curated list of top, high-volume stocks for quick analysis. Selecting an option here will automatically populate the custom symbol field."
    )

    manual_ticker_input_value = st.text_input(
        "Or Type Custom Symbol", 
        value=st.session_state.manual_ticker_input_value_key, 
        key="manual_ticker_input_widget", 
        help="Enter a custom stock ticker symbol (e.g., TSLA, MSFT). Data is fetched from Yahoo Finance. This field will update if you select a 'Top Stock'."
    ).upper()
    
    ticker = st.session_state.manual_ticker_input_value_key
    
    if not ticker and selected_top_stock:
        ticker = selected_top_stock
    elif not ticker:
        ticker = "NVDA"


    col1, col2 = st.columns(2)
    with col1:
        def update_period_options_callback():
            current_interval = st.session_state.selected_interval_key_widget
            valid_periods_for_current_interval = []
            if current_interval == "1m":
                valid_periods_for_current_interval = ["1d", "5d", "7d"] 
            elif current_interval in ["5m", "15m", "30m"]:
                valid_periods_for_current_interval = ["1d", "5d", "1mo", "2mo"] 
            else:
                valid_periods_for_current_interval = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            
            if st.session_state.selected_period_key_widget not in valid_periods_for_current_interval:
                if "3mo" in valid_periods_for_current_interval:
                    st.session_state.selected_period_key_widget = "3mo"
                elif valid_periods_for_current_interval:
                    current_period_index = 0
                    st.session_state.selected_period_key_widget = valid_periods_for_current_interval[current_period_index]
                else:
                    st.session_state.selected_period_key_widget = ""


        interval = st.selectbox(
            "Interval", 
            ["1m", "5m", "15m", "30m", "1h", "1d"], 
            key="selected_interval_key_widget", 
            on_change=update_period_options_callback, 
            help="Select the time interval for each candlestick (e.g., '1h' for hourly bars). This affects chart granularity. **Warning: 1-minute data is limited to ~7 days, and 5/15/30-minute data to ~60 days from Yahoo Finance.**"
        )
        
    with col2:
        current_interval_for_period = st.session_state.selected_interval_key_widget
        valid_periods = []
        if current_interval_for_period == "1m":
            valid_periods = ["1d", "5d", "7d"]
        elif current_interval_for_period in ["5m", "15m", "30m"]:
            valid_periods = ["1d", "5d", "1mo", "2mo"]
        else:
            valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        
        current_period_index = 0
        if st.session_state.selected_period_key_widget in valid_periods:
            current_period_index = valid_periods.index(st.session_state.selected_period_key_widget)
        else:
            if "3mo" in valid_periods:
                current_period_index = valid_periods.index("3mo")
            elif valid_periods:
                current_period_index = 0
            st.session_state.selected_period_key_widget = valid_periods[current_period_index] if valid_periods else ""


        # FIX: Removed the 'value' argument as its state is managed by 'key' and 'index'
        period = st.selectbox(
            "Period", 
            valid_periods, 
            index=current_period_index, 
            key="selected_period_key_widget", 
            help="Select the total historical period to fetch data for. This list dynamically adjusts based on your chosen interval due to data provider limitations. Longer periods with small intervals can be very slow or fail."
        )
    
    st.markdown("---")
    st.markdown("### 📊 Technical Indicators")
    show_rsi = st.checkbox("Show RSI", True, help="**Relative Strength Index (RSI)**: A momentum oscillator measuring the speed and change of price movements. Helps identify overbought (>70) or oversold (<30) conditions. Typically plotted below the main price chart.")
    
    if show_rsi:
        st.session_state.rsi_period = st.number_input("RSI Period", min_value=1, value=st.session_state.rsi_period, key="rsi_period_input", help="Number of periods to calculate RSI over (e.g., 14 for standard).")
    
    show_macd = st.checkbox("Show MACD", True, help="**Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator showing the relationship between two moving averages of a stock's price. A bullish signal occurs when MACD crosses above its signal line, and vice-versa for bearish. Often plotted as a histogram below the main price chart.")
    
    if show_macd:
        st.session_state.macd_fast_period = st.number_input("MACD Fast Period", min_value=1, value=st.session_state.macd_fast_period, key="macd_fast_period_input", help="Number of periods for the faster EMA (e.g., 12).")
        st.session_state.macd_slow_period = st.number_input("MACD Slow Period", min_value=1, value=st.session_state.macd_slow_period, key="macd_slow_period_input", help="Number of periods for the slower EMA (e.g., 26).")
        st.session_state.macd_signal_period = st.number_input("MACD Signal Period", min_value=1, value=st.session_state.macd_signal_period, key="macd_signal_period_input", help="Number of periods for the signal line EMA (e.g., 9).")
    
    show_vwap = st.checkbox("Show VWAP", True, help="**Volume Weighted Average Price (VWAP)**: The average price a security has traded at throughout the day, based on both volume and price. Traders often use it as a trend confirmation tool, with price above VWAP being bullish and below being bearish. Plotted directly on the main price chart.")
    show_sr = st.checkbox("Show S/R Levels", True, help="**Support and Resistance Levels**: Price points where a trend tends to pause or reverse due to concentrated supply or demand. Support is a price floor, Resistance is a price ceiling. Plotted as horizontal lines on the main price chart.")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; margin-top:1rem; opacity:0.8; color:var(--text-subtle);">
        <small>ℹ️ Data provided by Yahoo Finance and Finnhub.io</small>
    </div>
    """, unsafe_allow_html=True)

# --- Fetch OHLCV Data ---
@st.cache_data(ttl=60)
def get_stock_data(ticker, interval, period):
    try:
        df = yf.download(ticker, interval=interval, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            st.error(f" No data found for **{ticker}** with the selected interval/period. Please try different parameters or a valid ticker symbol.")
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df
    except Exception as e:
        error_msg = str(e)
        if "Broken pipe" in error_msg or "Errno 32" in error_msg:
            st.warning(f"⚠️ Connection issue with **{ticker}**. Please try again - this is usually a temporary network problem.")
        else:
            st.error(f"Error fetching data for **{ticker}**: {error_msg}. Please check the symbol or your internet connection.")
        return None

df = get_stock_data(ticker, interval, period)
if df is None or df.empty:
    st.stop()

# --- Improved VWAP Calculation ---
def calculate_vwap(df):
    df = df.copy()
    df.loc[:, 'TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df.loc[:, 'price_volume_weighted'] = df['TypicalPrice'] * df['Volume']
    
    df.loc[:, 'CumVol'] = df['Volume'].cumsum()
    df.loc[:, 'CumVolPrice'] = df['price_volume_weighted'].cumsum()
    
    df.loc[:, 'VWAP'] = df['CumVolPrice'] / df['CumVol'].replace(0, np.nan)
    
    df.loc[:, 'VWAP'] = df['VWAP'].ffill().bfill()

    df.drop(columns=['TypicalPrice', 'price_volume_weighted'], inplace=True, errors='ignore')
    
    return df

df = calculate_vwap(df)

# --- Improved Support/Resistance Calculation ---
def find_pivots(series, window=5):
    if window % 2 == 0:
        window += 1
    if window < 3: 
        window = 3

    min_val = series.rolling(window=window, center=True).min()
    max_val = series.rolling(window=window, center=True).max()
    
    supports_raw = series[series == min_val]
    resistances_raw = series[series == max_val]
    
    tolerance_factor = 0.005 

    unique_supports = []
    if not supports_raw.empty:
        sorted_supports = sorted(supports_raw.unique())
        if sorted_supports: 
            unique_supports.append(sorted_supports[0])
            for val in sorted_supports[1:]:
                threshold = unique_supports[-1] * tolerance_factor if unique_supports[-1] != 0 else series.mean() * tolerance_factor
                if (abs(val - unique_supports[-1]) > threshold):
                    unique_supports.append(val)
                else: 
                    unique_supports[-1] = (unique_supports[-1] + val) / 2
            
    unique_resistances = []
    if not resistances_raw.empty:
        sorted_resistances = sorted(resistances_raw.unique(), reverse=True)
        if sorted_resistances: 
            unique_resistances.append(sorted_resistances[0])
            for val in sorted_resistances[1:]:
                threshold = unique_resistances[-1] * tolerance_factor if unique_resistances[-1] != 0 else series.mean() * tolerance_factor
                if (abs(val - unique_resistances[-1]) > threshold):
                    unique_resistances.append(val)
                else: 
                    unique_resistances[-1] = (unique_resistances[-1] + val) / 2
                
    return sorted(unique_supports), sorted(unique_resistances, reverse=True)

support_levels, resistance_levels = find_pivots(df['Close'], window=10)

# --- Technical Indicators ---
def calculate_technical_indicators(df, rsi_period, macd_fast_period, macd_slow_period, macd_signal_period):
    close_prices = df['Close']
    
    # RSI
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9) 
    
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50) 
    
    # MACD
    ema_fast = close_prices.ewm(span=macd_fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=macd_slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal_period, adjust=False).mean()
    
    return rsi, macd_line, signal_line

rsi, macd, signal = calculate_technical_indicators(df, st.session_state.rsi_period, st.session_state.macd_fast_period, st.session_state.macd_slow_period, st.session_state.macd_signal_period)

# --- Current Metrics ---
if len(df) < 2:
    current_price = df['Close'].iloc[-1] if not df.empty else 0.0
    prev_close = current_price 
    price_change = 0.0
    percent_change = 0.0
    vwap_value = df['VWAP'].iloc[-1] if 'VWAP' in df.columns and not df['VWAP'].isnull().all() else current_price
    vwap_diff = ((current_price - vwap_value) / (vwap_value + 1e-9)) * 100 
    st.warning("Not enough historical data for full metric calculation. Displaying available data.") 
else:
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    price_change = current_price - prev_close
    percent_change = (price_change / prev_close) * 100 if prev_close != 0 else 0.0
    vwap_value = df['VWAP'].iloc[-1] if 'VWAP' in df.columns and not df['VWAP'].isnull().all() else current_price
    vwap_diff = ((current_price - vwap_value) / (vwap_value + 1e-9)) * 100


with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box glow-box"><h4>Current Price</h4><h3>${current_price:.2f}</h3></div>', unsafe_allow_html=True)
    with col2:
        change_class = "positive" if price_change >= 0 else "negative"
        st.markdown(f'<div class="metric-box"><h4>Price Change</h4><h3 class="{change_class}">{price_change:.2f} ({percent_change:.2f}%)</h3></div>', unsafe_allow_html=True)
    with col3:
        volume_display = df['Volume'].iloc[-1] if not df['Volume'].empty else 0
        st.markdown(f'<div class="metric-box"><h4>Volume</h4><h3>{volume_display:,.0f}</h3></div>', unsafe_allow_html=True)
    with col4:
        vwap_class = "positive" if vwap_diff >= 0 else "negative"
        st.markdown(f'<div class="metric-box"><h4>VWAP Diff</h4><h3 class="{vwap_class}">{vwap_diff:.2f}%</h3></div>', unsafe_allow_html=True)

# --- Company Profile / Key Fundamentals ---
st.markdown('<div class="section"><h3>Company Overview</h3></div>', unsafe_allow_html=True)
try:
    stock_info = yf.Ticker(ticker).info
    company_name = stock_info.get('longName', 'N/A')
    sector = stock_info.get('sector', 'N/A')
    industry = stock_info.get('industry', 'N/A')
    market_cap = stock_info.get('marketCap', 'N/A')
    pe_ratio = stock_info.get('trailingPE', 'N/A')
    dividend_yield = stock_info.get('dividendYield', 'N/A')

    st.markdown(f"""
    <div class="metric-box" style="padding:1rem;">
        <p><b>Company Name:</b> {company_name}</p>
        <p><b>Sector:</b> {sector}</p>
        <p><b>Industry:</b> {industry}</p>
        <p><b>Market Cap:</b> ${market_cap:,.0f} (approx)</p>
        <p><b>Trailing P/E:</b> {pe_ratio:.2f}</p>
        <p><b>Dividend Yield:</b> {(dividend_yield*100):.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not fetch company overview for {ticker}: {e}", icon="⚠️")

# --- Top Stocks Overview Section (New) ---
st.markdown('<div class="section"><h2>⚡ Top Market Insights at a Glance</h2></div>', unsafe_allow_html=True)

# Extended list of top stocks, focusing on high-volume, well-known ones for better data reliability
OVERVIEW_STOCKS = list(set(TOP_STOCKS + ["SMCI", "GOOG", "AMZN", "MSFT", "TSM", "ASML", "CRM", "ADBE", "INTU", "ORCL", "COST"]))
OVERVIEW_STOCKS = OVERVIEW_STOCKS[:15]

@st.cache_data(ttl=5 * 60)
def get_overview_data(tickers, api_key, _sentiment_analyzer):
    overview_results = []
    progress_bar = st.progress(0, text="Analyzing top stocks for quick insights...")

    for i, tkr in enumerate(tickers):
        score_checks = 0
        criteria_details = {}

        try:
            df_ohlcv = yf.download(tkr, period="5d", interval="1d", progress=False, auto_adjust=True)
            if df_ohlcv.empty or len(df_ohlcv) < 2:
                raise ValueError("Insufficient OHLCV data.")

            current_price = df_ohlcv['Close'].iloc[-1]
            prev_close = df_ohlcv['Close'].iloc[-2]
            price_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0

            if price_change_pct > 0:
                criteria_details['Price Up Today'] = "✅"
                score_checks += 1
            else:
                criteria_details['Price Up Today'] = "❌"

            current_volume = df_ohlcv['Volume'].iloc[-1]
            avg_volume = df_ohlcv['Volume'].iloc[:-1].mean()
            if current_volume > (avg_volume * 1.2):
                criteria_details['High Volume'] = "✅"
                score_checks += 1
            else:
                criteria_details['High Volume'] = "❌"

            temp_df_vwap = df_ohlcv.copy()
            temp_df_vwap['TypicalPrice'] = (temp_df_vwap['High'] + temp_df_vwap['Low'] + temp_df_vwap['Close']) / 3
            temp_df_vwap['price_volume_weighted'] = temp_df_vwap['TypicalPrice'] * temp_df_vwap['Volume']
            temp_df_vwap['CumVol'] = temp_df_vwap['Volume'].cumsum()
            temp_df_vwap['CumVolPrice'] = temp_df_vwap['price_volume_weighted'].cumsum()
            temp_df_vwap['VWAP'] = temp_df_vwap['CumVolPrice'] / temp_df_vwap['CumVol'].replace(0, np.nan)
            
            latest_vwap = temp_df_vwap['VWAP'].iloc[-1] if 'VWAP' in temp_df_vwap.columns and not temp_df_vwap['VWAP'].isnull().all() else current_price
            if current_price > latest_vwap:
                criteria_details['Above VWAP'] = "✅"
                score_checks += 1
            else:
                criteria_details['Above VWAP'] = "❌"

            delta = df_ohlcv['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(span=14, adjust=False).mean()
            avg_loss = loss.ewm(span=14, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-9) 
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50
            if 30 <= latest_rsi <= 70:
                criteria_details['Healthy RSI (30-70)'] = "✅"
                score_checks += 1
            else:
                criteria_details['Healthy RSI (30-70)'] = "❌"

            ema12 = df_ohlcv['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df_ohlcv['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            latest_macd = macd_line.iloc[-1] if not macd_line.empty and pd.notna(macd_line.iloc[-1]) else 0
            latest_signal = signal_line.iloc[-1] if not signal_line.empty and pd.notna(signal_line.iloc[-1]) else 0

            if latest_macd > latest_signal:
                criteria_details['MACD Bullish Cross'] = "✅"
                score_checks += 1
            else:
                criteria_details['MACD Bullish Cross'] = "❌"

            news_articles = get_news_sentiment_overview(tkr, api_key)
            sentiment_df = analyze_sentiment_for_articles_vader(news_articles, _sentiment_analyzer)
            
            avg_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0
            if avg_sentiment > 0.1:
                criteria_details['Positive News Sentiment'] = "✅"
                score_checks += 1
            else:
                criteria_details['Positive News Sentiment'] = "❌"


            overview_results.append({
                "Symbol": tkr,
                "Score": f"{score_checks}/6", 
                "Overall Score": score_checks,
                **criteria_details
            })

        except Exception as e: # Catch all exceptions for robustness
            overview_results.append({
                "Symbol": tkr,
                "Score": "0/6",
                "Overall Score": 0,
                'Price Up Today': "❌",
                'High Volume': "❌",
                'Above VWAP': "❌",
                'Healthy RSI (30-70)': "❌",
                'MACD Bullish Cross': "❌",
                'Positive News Sentiment': "❌",
            })
        progress_bar.progress((i + 1) / len(tickers), text=f"Analyzing top stocks: {i+1}/{len(tickers)}")
    progress_bar.empty()
    return pd.DataFrame(overview_results)

@st.cache_data(ttl=300)
def get_news_sentiment_overview(ticker, api_key):
    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={API_KEY}"
    try:
        news_response = requests.get(news_url)
        news_response.raise_for_status()
        return news_response.json()
    except requests.exceptions.RequestException:
        return []
    except ValueError:
        return []

@st.cache_resource
def load_sentiment_analyzer_global():
    with st.spinner("Loading global sentiment analyzer..."):
        try:
            # Check if lexicon is available
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            # Download if not found - handle SSL issues
            try:
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                nltk.download('vader_lexicon', quiet=True)
            except Exception as e:
                st.warning(f"Could not download vader_lexicon automatically: {e}. Please run: python3 -m nltk.downloader vader_lexicon")
                return None
        # Initialize analyzer
        try:
            return SentimentIntensityAnalyzer()
        except Exception as e:
            st.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
            return None
global_sentiment_analyzer = load_sentiment_analyzer_global()


overview_df = get_overview_data(OVERVIEW_STOCKS, API_KEY, global_sentiment_analyzer)

if not overview_df.empty:
    overview_df = overview_df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
    st.markdown("""
        <p style='color:var(--text-subtle); font-size:0.9rem;'>
            A quick glance at top stocks based on current performance and market signals.
            **✅ Indicates a positive signal for that criterion, ❌ indicates a neutral or negative signal.**
            Higher scores (e.g., 5/6, 6/6) suggest stronger overall positive momentum.
        </p>
    """, unsafe_allow_html=True)
    st.dataframe(
        overview_df[['Symbol', 'Score', 'Price Up Today', 'High Volume', 'Above VWAP', 'Healthy RSI (30-70)', 'MACD Bullish Cross', 'Positive News Sentiment']],
        use_container_width=True
    )
else:
    st.warning("Could not generate the Top Stocks Overview. This might be due to API limits, network issues, or lack of data for all tickers. Try refreshing in a few minutes.", icon="⚠️")

# --- Main Tabs (Re-ordered and added Watchlist) ---
# Current Tabs: "📊 Live Charts", "📰 Market Pulse", "🔄 Options Flow", "💰 Dividend Analysis", "📚 Learning Center", "⭐ My Watchlist"
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Live Charts", "📰 Market Pulse", "🔄 Options Flow", "💰 Dividend Analysis", "📚 Learning Center", "⭐ My Watchlist"])

# --- Tab 1: Live Charts ---
with tab1:
    st.markdown('<div class="section"><h2>Market Chart Analysis</h2></div>', unsafe_allow_html=True)
    
    entry_signal = ""
    if len(df) > 1 and not df["VWAP"].isnull().all() and not df['Close'].isnull().all():
        # FIX: Corrected the unterminated string literal
        if pd.notna(df["VWAP"].iloc[-2]) and pd.notna(df['Close'].iloc[-2]):
            if df['Close'].iloc[-1] > df["VWAP"].iloc[-1] and (df['Close'].iloc[-2] <= df["VWAP"].iloc[-2]):
                entry_signal = "📈 Market Signal: Price crossed above VWAP (bullish momentum detected)"
            elif df['Close'].iloc[-1] < df["VWAP"].iloc[-1] and (df['Close'].iloc[-2] >= df["VWAP"].iloc[-2]):
                entry_signal = "📉 Market Signal: Price dropped below VWAP (bearish momentum detected)"
    
    if entry_signal:
        st.info(entry_signal, icon="🚨")
    
    st.markdown("""
        <p style='color:var(--text-subtle); font-size:0.9rem;'>
            The chart above displays candlestick patterns, representing price movements over selected intervals.
            Green candles indicate price increased, red indicate price decreased.
        </p>
    """, unsafe_allow_html=True)

    plot_df = df.tail(100).copy()
    plot_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    
    if plot_df.empty:
        st.warning("Not enough valid data points to render the chart. Try a different interval or period.")
        st.stop()

    # --- Plotly Interactive Chart ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=plot_df.index,
                                    open=plot_df['Open'],
                                    high=plot_df['High'],
                                    low=plot_df['Low'],
                                    close=plot_df['Close'],
                                    name='Candlesticks',
                                    increasing_line_color=SUCCESS_COLOR_HEX,
                                    decreasing_line_color=DANGER_COLOR_HEX),
                    row=1, col=1)

    # Add VWAP
    if show_vwap and 'VWAP' in plot_df.columns and not plot_df['VWAP'].isnull().all():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], mode='lines',
                                    name='VWAP', line=dict(color=PRIMARY_ACCENT_COLOR_HEX, width=1.5)),
                        row=1, col=1)

    # Add S/R Levels (Plotly version)
    if show_sr and (support_levels or resistance_levels):
        for level in support_levels:
            fig.add_hline(y=level, line_color=SUCCESS_COLOR_HEX, line_dash="dot", line_width=1,
                            annotation_text=f"Support {level:.2f}", annotation_position="bottom right",
                            row=1, col=1)
        for level in resistance_levels:
            fig.add_hline(y=level, line_color=DANGER_COLOR_HEX, line_dash="dot", line_width=1,
                            annotation_text=f"Resistance {level:.2f}", annotation_position="top right",
                            row=1, col=1)

    # Add Volume
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Volume',
                            marker_color=INFO_COLOR_HEX, opacity=0.7),
                    row=2, col=1)

    # Update layout and axis styling for Plotly
    fig.update_layout(
        template="plotly_dark", # Use a dark theme template
        xaxis_rangeslider_visible=False, # Hide the bottom range slider
        height=600,
        margin=dict(l=20, r=20, t=20, b=20), # Adjust margins for cleaner look
        paper_bgcolor=BG_DARK_COLOR_HEX, # Match Streamlit app background
        plot_bgcolor=BG_DARK_COLOR_HEX,
        font=dict(color=TEXT_LIGHT_COLOR_HEX), # Global font color
        hovermode="x unified", # Shows all data points at a given X coordinate on hover
    )
    
    # Update axes to match your custom CSS grid/border colors
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR_FOR_MPL_PLOTLY, # Corrected to use direct hex variable
        zeroline=False,
        showline=True, linewidth=1, linecolor=BORDER_COLOR_FOR_MPL_PLOTLY,
        tickfont=dict(color=TEXT_LIGHT_COLOR_HEX),
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor=BORDER_COLOR_FOR_MPL_PLOTLY, # Corrected to use direct hex variable
        zeroline=False,
        showline=True, linewidth=1, linecolor=BORDER_COLOR_FOR_MPL_PLOTLY,
        tickfont=dict(color=TEXT_LIGHT_COLOR_HEX),
        row=1, col=1, title_text="Price"
    )
    fig.update_yaxes(
        tickformat=".2s", # Format volume ticks (e.g., 100M, 1B)
        row=2, col=1, title_text="Volume",
        tickfont=dict(color=TEXT_LIGHT_COLOR_HEX),
    )

    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section"><h3>Momentum Scanner</h3></div>', unsafe_allow_html=True)
        st.markdown("""
            <p style='color:var(--text-subtle); font-size:0.9rem;'>
                The **Relative Strength Index (RSI)**: A momentum oscillator measuring the speed and change of price movements. 
                RSI above 70 suggests overbought conditions (potential pullback), while below 30 suggests oversold conditions (potential bounce).
            </p>
        """, unsafe_allow_html=True)
        latest_rsi = rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50
        rsi_status = "Overbought (>70)" if latest_rsi > 70 else "Oversold (<30)" if latest_rsi < 30 else "Neutral"
        rsi_class = "negative" if latest_rsi > 70 else "positive" if latest_rsi < 30 else "neutral"
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>RSI ({st.session_state.rsi_period}): <span class="{rsi_class}">{latest_rsi:.2f}</span></h4>
            <p>Status: <strong>{rsi_status}</strong></p>
            <progress value="{latest_rsi}" max="100"></progress>
        </div>
        """, unsafe_allow_html=True)
        
        if latest_rsi < 30:
            st.success("Market analysis detects oversold conditions - potential reversal opportunity", icon="📉")
        elif latest_rsi > 70:
            st.error("Market analysis detects overbought conditions - potential pullback opportunity", icon="📈")
        else:
            st.info("Momentum in equilibrium - monitor for breakout signals", icon="⚖️")
    
    with col2:
        st.markdown('<div class="section"><h3>Trend Analysis</h3></div>', unsafe_allow_html=True)
        st.markdown("""
            <p style='color:var(--text-subtle); font-size:0.9rem;'>
                **MACD** (Moving Average Convergence Divergence): A trend-following momentum indicator showing the relationship between two moving averages of a stock's price.
                A bullish signal occurs when the MACD line crosses above its signal line, and vice-versa for bearish.
            </p>
        """, unsafe_allow_html=True)
        latest_macd = macd.iloc[-1] if not macd.empty and pd.notna(macd.iloc[-1]) else 0.0
        latest_signal = signal.iloc[-1] if not signal.empty and pd.notna(signal.iloc[-1]) else 0.0
        macd_diff = latest_macd - latest_signal
        macd_trend = "Bullish" if macd_diff > 0 else "Bearish"
        macd_class = "positive" if macd_diff > 0 else "negative"
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>MACD: <span class="{macd_class}">{latest_macd:.2f}</span></h4>
            <h4>Signal Line: {latest_signal:.2f}</h4>
            <p>Market Trend: <strong class="{macd_class}">{macd_trend}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if macd_diff > 0 and macd_diff > 0.05:
            st.success("Strong bullish momentum detected", icon="🚀")
        elif macd_diff < 0 and macd_diff < -0.05:
            st.error("Strong bearish momentum detected", icon="🛑")
        else:
            st.info("Trend momentum neutral - awaiting confirmation signals", icon="⚖️")
    
    if show_sr and (support_levels or resistance_levels):
        st.markdown('<div class="section"><h3>Key Levels</h3></div>', unsafe_allow_html=True)
        st.markdown("""
            <p style='color:var(--text-subtle); font-size:0.9rem;'>
                **Support and Resistance (S/R) levels**: Price points where a trend tends to pause or reverse due to concentrated supply or demand.
                Support acts as a price floor, while Resistance acts as a price ceiling.
            </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🛡️ Support Matrix")
            if support_levels:
                below_current = [s for s in support_levels if s <= current_price]
                above_current = [s for s in support_levels if s > current_price]

                for level in sorted(below_current, reverse=True)[:5]:
                    diff_pct = ((current_price - level) / (level + 1e-9)) * 100 
                    st.markdown(f"""
                    <div class="sr-level support">
                        <strong>${level:.2f}</strong> <span>({abs(diff_pct):.2f}% below current)</span>
                    </div>
                    """, unsafe_allow_html=True)
                if below_current and above_current:
                    st.markdown("<hr style='border-top: 1px dashed var(--divider-color); margin: 1rem 0;'>", unsafe_allow_html=True)
                for level in sorted(above_current)[:5]:
                    diff_pct = ((current_price - level) / (level + 1e-9)) * 100 
                    st.markdown(f"""
                    <div class="sr-level support">
                        <strong>${level:.2f}</strong> <span>({abs(diff_pct):.2f}% above current)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant support levels detected.", icon="🤷")
        
        with col2:
            st.markdown("##### 🚀 Resistance Matrix")
            if resistance_levels:
                above_current_res = [r for r in resistance_levels if r >= current_price]
                below_current_res = [r for r in resistance_levels if r < current_price]

                for level in sorted(above_current_res)[:5]:
                    diff_pct = ((level - current_price) / (current_price + 1e-9)) * 100
                    st.markdown(f"""
                    <div class="sr-level resistance">
                        <strong>${level:.2f}</strong> <span>({diff_pct:.2f}% above current)</span>
                    </div>
                    """, unsafe_allow_html=True)
                if above_current_res and below_current_res:
                    st.markdown("<hr style='border-top: 1px dashed var(--divider-color); margin: 1rem 0;'>", unsafe_allow_html=True)
                for level in sorted(below_current_res, reverse=True)[:5]:
                    diff_pct = ((level - current_price) / (current_price + 1e-9)) * 100
                    st.markdown(f"""
                    <div class="sr-level resistance">
                        <strong>${level:.2f}</strong> <span>({abs(diff_pct):.2f}% below current)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant resistance levels detected.", icon="🤷")

# --- Tab 2: Market Pulse ---
with tab2:
    st.markdown('<div class="section"><h2>Market Pulse</h2></div>', unsafe_allow_html=True)
    
    @st.cache_data(ttl=3600)
    def get_news(ticker):
        today = datetime.now(timezone.utc)
        from_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={API_KEY}"
        try:
            news_response = requests.get(news_url)
            news_response.raise_for_status()
            return news_response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching news from Finnhub: {e}")
            return None
        except ValueError:
            return None
    
    @st.cache_resource
    def load_sentiment_analyzer():
        with st.spinner("Loading sentiment analyzer..."):
            try:
                # Check if lexicon is available
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                # Download if not found - handle SSL issues
                try:
                    import ssl
                    try:
                        _create_unverified_https_context = ssl._create_unverified_context
                    except AttributeError:
                        pass
                    else:
                        ssl._create_default_https_context = _create_unverified_https_context
                    nltk.download('vader_lexicon', quiet=True)
                except Exception as e:
                    st.warning(f"Could not download vader_lexicon automatically: {e}. Please run: python3 -m nltk.downloader vader_lexicon")
                    return None
            # Initialize analyzer
            try:
                return SentimentIntensityAnalyzer()
            except Exception as e:
                st.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
                return None

    sentiment_analyzer = load_sentiment_analyzer()

    def analyze_sentiment_for_articles_vader(articles, analyzer):
        sentiment_data = []
        if not articles:
            return pd.DataFrame()

        for article in articles:
            text_to_analyze = f"{article.get('headline', '')}. {article.get('summary', '')}"
            if text_to_analyze.strip():
                try:
                    vs = analyzer.polarity_scores(text_to_analyze)
                    compound_score = vs['compound']
                    
                    sentiment_label = "NEUTRAL"
                    if compound_score >= 0.05:
                        sentiment_label = "POSITIVE"
                    elif compound_score <= -0.05:
                        sentiment_label = "NEGATIVE"
                    
                    # Corrected: Use datetime.fromtimestamp with timezone.utc
                    sentiment_data.append({
                        'date': datetime.fromtimestamp(article.get('datetime', 0), tz=timezone.utc).date(),
                        'sentiment_score': compound_score,
                        'sentiment_label': sentiment_label,
                        'headline': article.get('headline', '')
                    })
                except Exception as e:
                    st.warning(f"Could not analyze sentiment for an article (possibly malformed): {e}", icon="⚠️")
                    continue
        return pd.DataFrame(sentiment_data)

    articles = get_news(ticker)
    
    st.markdown('<div class="section"><h3>News Sentiment</h3></div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='color:var(--text-subtle); font-size:0.9rem;'>
            News sentiment analysis provides insight into the general tone of recent news articles related to the stock.
            Scores range from -1 (very negative) to +1 (very positive).
        </p>
    """, unsafe_allow_html=True)

    sentiment_df = pd.DataFrame()
    if articles:
        sentiment_df = analyze_sentiment_for_articles_vader(articles, sentiment_analyzer)

    if not sentiment_df.empty:
        daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['Date', 'Average Sentiment']
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_sentiment['Date'], daily_sentiment['Average Sentiment'], color=PRIMARY_ACCENT_COLOR_HEX)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax.fill_between(daily_sentiment['Date'], 0, daily_sentiment['Average Sentiment'], 
                        where=(daily_sentiment['Average Sentiment'] >= 0), color=SUCCESS_COLOR_HEX, alpha=0.3, interpolate=True)
        ax.fill_between(daily_sentiment['Date'], 0, daily_sentiment['Average Sentiment'], 
                        where=(daily_sentiment['Average Sentiment'] < 0), color=DANGER_COLOR_HEX, alpha=0.3, interpolate=True)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Average News Sentiment for {ticker}", color=TEXT_LIGHT_COLOR_HEX)
        ax.set_xlabel("Date", color=TEXT_LIGHT_COLOR_HEX)
        ax.set_ylabel("Sentiment Score", color=TEXT_LIGHT_COLOR_HEX)
        ax.tick_params(axis='x', colors=TEXT_LIGHT_COLOR_HEX)
        ax.tick_params(axis='y', colors=TEXT_LIGHT_COLOR_HEX)
        ax.spines['bottom'].set_color(BORDER_COLOR_FOR_MPL_PLOTLY) # Corrected
        ax.spines['left'].set_color(BORDER_COLOR_FOR_MPL_PLOTLY) # Corrected
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor(BG_DARK_COLOR_HEX)
        fig.patch.set_facecolor(BG_DARK_COLOR_HEX)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("Individual News Sentiment & Articles")
        st.dataframe(sentiment_df[['date', 'sentiment_label', 'sentiment_score', 'headline']], use_container_width=True)
    else:
        st.info(f"No sentiment data could be generated for **{ticker}** in the selected period. This might be due to no news found or an issue with the Finnhub API key.", icon="📊")

    st.markdown("<hr style='border-top: 1px dashed var(--divider-color); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section"><h3>Latest Market News</h3></div>', unsafe_allow_html=True)
    
    display_articles = [a for a in articles if a.get('headline') and a.get('summary')] if articles else []
    display_articles = display_articles[:5]

    if display_articles:
        for article in display_articles:
            with st.container():
                st.markdown(f"""
                <div class="news-card">
                    <h4>{article['headline']}</h4>
                    <small style="opacity:0.7;">{datetime.fromtimestamp(article.get('datetime', 0), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} • {article.get('source', 'Unknown')}</small>
                    <p style="margin:0.75rem 0;'>{article['summary']}</p>
                    <a href="{article.get('url', '#')}" target="_blank" style="color:var(--primary-accent); text-decoration:none;">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
    else: 
        st.info(f"No recent news articles with headlines/summaries found for **{ticker}** in the selected period. Check your Finnhub API key and ticker symbol.", icon="📰")

# --- Tab 3: Options Flow ---
# Function to calculate Profit/Loss for a single option leg at expiration (MOVED TO GLOBAL SCOPE)
def calculate_option_pnl(option_type, strike_price, premium, underlying_prices):
    pnl = []
    if option_type == 'call':
        for price in underlying_prices:
            profit_loss = max(0, price - strike_price) - premium
            pnl.append(profit_loss)
    elif option_type == 'put':
        for price in underlying_prices:
            profit_loss = max(0, strike_price - price) - premium
            pnl.append(profit_loss)
    return pnl

# Function to calculate Probability of Profit (POP) (NEW)
def calculate_pop(option_type, strike_price, premium, current_price, implied_volatility, dte):
    if dte <= 0 or implied_volatility <= 0:
        return np.nan
    
    time_to_expiration_years = dte / 365.0
    
    # Calculate the break-even price
    if option_type == 'call':
        breakeven_price = strike_price + premium
    elif option_type == 'put':
        breakeven_price = strike_price - premium
    else:
        return np.nan

    # Using Black-Scholes d2 for probability of expiring in the money
    # For POP (probability of profit), we need the probability of the price being beyond the breakeven.
    # This usually corresponds to N(d2) for calls and N(-d2) for puts in the Black-Scholes context,
    # but the exact application for "profit" needs careful setup of d1/d2.
    # A simpler approximation for POP for long options:
    # Probability that price ends up above (for call) or below (for put) the breakeven point.
    
    # Let's re-calculate d1 and d2 for the breakeven price as the target
    # Assuming risk-free rate (r) of 0 for simplicity, and dividend yield (q) of 0.
    r = 0.01 # A small, non-zero risk-free rate (e.g., 1%)
    q = 0.0  # Dividend yield
    
    d1_pop = (np.log(current_price / breakeven_price) + (r - q + 0.5 * implied_volatility**2) * time_to_expiration_years) / (implied_volatility * np.sqrt(time_to_expiration_years))
    d2_pop = d1_pop - implied_volatility * np.sqrt(time_to_expiration_years)

    if option_type == 'call':
        pop = norm.cdf(d2_pop)
    elif option_type == 'put':
        pop = norm.cdf(-d2_pop)
    else:
        pop = np.nan
        
    return pop


# Function to analyze option (MOVED TO GLOBAL SCOPE)
def analyze_option(row, current_price, dte, option_type): # Added option_type
    price = row["lastPrice"]
    iv = row["impliedVolatility"]
    volume, oi = row["volume"], row["openInterest"]
    distance = abs(row["strike"] - current_price) 
    delta = row.get("delta", 0.0) if pd.notnull(row.get("delta", 0.0)) else 0.0
    delta_score = 1 - abs(abs(delta) - 0.5)

    reasons = []
    score_count = 0

    if delta_score > 0.7: 
        reasons.append("✅ Optimal delta for directional plays (closer to 0.5)")
        score_count += 1
    else:
        reasons.append("⚠️ Delta suggests limited price sensitivity or aggressive positioning")

    if volume > 1000: 
        reasons.append("✅ Strong liquidity with high volume (over 1,000 contracts)")
        score_count += 1
    else:
        reasons.append("⚠️ Low volume may impact execution (below 1,000 contracts)")

    if oi > 1000: 
        reasons.append("✅ High open interest indicates market conviction (over 1,000 contracts)")
        score_count += 1
    else:
        reasons.append("⚠️ Low open interest suggests weak participation (below 1,000 contracts)")

    if 0.5 < iv < 1.5: 
        reasons.append("✅ Implied Volatility within optimal range (0.5 - 1.5)")
        score_count += 1
    elif iv >= 1.5:
        reasons.append("⚠️ High Implied Volatility may indicate expensive premiums or extreme volatility")
    else:
        reasons.append("⚠️ Low Implied Volatility may indicate limited price movement expected")

    if distance < current_price * 0.05: 
        reasons.append(f"✅ Strike near current price (within 5% of ${current_price:.2f}) for higher probability")
        score_count += 1
    else:
        reasons.append(f"⚠️ Distant strike (further than 5% from ${current_price:.2f}) reduces probability of profit")
    
    stop_loss = round(price * 0.80, 2) 
    target_price = round(price * 1.50, 2) 

    risk = price - stop_loss
    reward = target_price - price
    
    if risk > 0 and reward > 0: 
        rr_ratio = reward / risk 
    else:
        rr_ratio = float('nan')

    pop = calculate_pop(option_type, row['strike'], row['lastPrice'], current_price, iv, dte) # Pass option_type

    suggestion = "🚀 High-Probability Trade" if score_count >= 4 else \
                                "⚠️ Caution Advised" if score_count >= 2 else \
                                "🛑 High-Risk Setup"

    return suggestion, reasons, stop_loss, target_price, rr_ratio, pop


with tab3:
    st.markdown('<div class="section"><h2>Options Flow</h2></div>', unsafe_allow_html=True)
    
    @st.cache_resource(ttl=300)
    def get_options_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                st.warning(f"No historical data available for **{ticker}** to determine current price for options analysis.")
                return None
            current_price = hist["Close"].iloc[-1]
            expirations = stock.options
            return {
                'current_price': current_price,
                'expirations': expirations,
                'ticker': ticker
            }
        except Exception as e:
            st.error(f"Options scan failed: {str(e)}")
            return None
    
    options_data = get_options_data(ticker)
    
    if not options_data or not options_data['expirations']:
        st.warning("Scan detected no options flow for this symbol.", icon="🔍")
    else:
        current_price = options_data['current_price']
        expirations = options_data['expirations']
        
        # Function to rank options
        def rank_options(df_original, is_call=True):
            df = df_original.copy()
            # Ensure 'impliedVolatility' is a column name; yfinance provides 'impliedVolatility' as a direct column.
            # Handle cases where columns might be missing or all NaN for a given chain.
            required_cols = ["volume", "openInterest", "impliedVolatility", "bid", "ask", "strike", "lastPrice"]
            df.dropna(subset=required_cols, inplace=True)
            
            if df.empty: 
                return pd.DataFrame() 

            df.loc[:, "spread"] = (df["ask"] - df["bid"]).abs()
            df.loc[:, "distance_to_strike"] = (df["strike"] - current_price).abs()
            
            # Ensure 'delta' column exists before trying to access it
            if 'delta' in df.columns:
                df.loc[:, "delta"] = df["delta"].fillna(0.5) # Fill NaNs with a neutral delta
            else:
                # If delta is not available from yfinance, use a sensible default or skip delta_score
                df.loc[:, "delta"] = 0.5 
            df.loc[:, "delta_score"] = df["delta"].apply(lambda x: 1 - abs(abs(x) - 0.5)) 
            
            # Handle potential division by zero for max values
            max_volume = df["volume"].max() if df["volume"].max() > 0 else 1e-9
            max_openInterest = df["openInterest"].max() if df["openInterest"].max() > 0 else 1e-9
            max_iv = df["impliedVolatility"].max() if df["impliedVolatility"].max() > 0 else 1e-9
            max_spread = df["spread"].max() if df["spread"].max() > 0 else 1e-9
            max_distance = df["distance_to_strike"].max() if df["distance_to_strike"].max() > 0 else 1e-9

            df.loc[:, "Score"] = (
                (df["volume"] / max_volume) * 0.3 +
                (df["openInterest"] / max_openInterest) * 0.25 +
                (1 - (df["impliedVolatility"] - 0.4).abs() / max_iv) * 0.15 + # Calculate distance from 0.4 IV
                (1 - df["spread"] / max_spread) * 0.1 + 
                (1 - df["distance_to_strike"] / max_distance) * 0.1 + 
                df["delta_score"] * 0.1
            )
            return df.sort_values("Score", ascending=False).head(10)


        exp_date_str = st.selectbox("Select Expiration", expirations, key="exp_date", help="Choose an expiration date for options contracts. This determines the expiry of the calls and puts listed below.")
        
        # Calculate DTE for POP calculation
        current_utc_date = datetime.now(timezone.utc).date()
        expiration_date = pd.to_datetime(exp_date_str).date()
        dte_timedelta = expiration_date - current_utc_date
        dte = dte_timedelta.days # Get days as integer

        stock = yf.Ticker(ticker)
        chain = stock.option_chain(exp_date_str)
        calls, puts = chain.calls.copy(), chain.puts.copy()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>📈 Call Flow</h3>", unsafe_allow_html=True)
            st.markdown("""
                <p style='color:var(--text-subtle); font-size:0.9rem;'>
                    Call options give the holder the right, but not the obligation, to *buy* a stock at a specified strike price.
                    They are generally bought when expecting the stock price to rise.
                </p>
            """, unsafe_allow_html=True)
            top_calls = rank_options(calls, True)
            if not top_calls.empty:
                top_calls['display_name'] = top_calls['contractSymbol'] + ' - $' + top_calls['strike'].astype(str) + ' (IV: ' + (top_calls['impliedVolatility']*100).round(2).astype(str) + '%)'
                call_selection = st.selectbox("Select Call", top_calls["display_name"], key="call_select", help="Select a call option from the ranked list based on liquidity, implied volatility, and proximity to current price.")
                st.dataframe(top_calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'Score']], 
                                 use_container_width=True)
                
                selected_call_row = top_calls[top_calls["display_name"] == call_selection].iloc[0]
                # Pass 'call' as the option_type
                suggestion, reasons, stop, target, rr_ratio, pop = analyze_option(selected_call_row, current_price, dte, 'call')

                st.markdown("<h3>🧠 Analysis</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-box">
                    <p><b>Contract:</b> {selected_call_row['contractSymbol']}</p>
                    <p><b>Strike:</b> ${selected_call_row['strike']:.2f}</p>
                    <p><b>Last:</b> ${selected_call_row['lastPrice']:.2f}</p>
                    <p><b>Spread:</b> ${selected_call_row['bid']:.2f} / ${selected_call_row['ask']:.2f}</p>
                    <p><b>IV:</b> {selected_call_row['impliedVolatility']:.2%}</p>
                    <p><b>Probability of Profit (POP):</b> {(pop*100):.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h4>📊 Trade Setup</h4>", unsafe_allow_html=True)
                rr_display = f"{rr_ratio:.1f}" if not pd.isna(rr_ratio) else "N/A"
                st.markdown(f"""
                - 🎯 Target: ${target:.2f} ({(target/selected_call_row['lastPrice']-1)*100:.0f}%)
                - 🛑 Stop: ${stop:.2f}
                - ⚖️ R/R: {rr_display}
                """)
                
                st.markdown("<h4>🔍 Insights</h4>", unsafe_allow_html=True)
                for r in reasons:
                    st.markdown(f"- {r}")
                
                if "High-Probability" in suggestion:
                    st.success(suggestion, icon="🚀")
                elif "Caution" in suggestion:
                    st.warning(suggestion, icon="⚠️")
                else:
                    st.error(suggestion, icon="🛑")

                # Expected Move Calculation & Display for selected option
                try:
                    atm_iv = selected_call_row['impliedVolatility']
                    
                    if dte > 0 and atm_iv > 0:
                        expected_move = current_price * atm_iv * np.sqrt(dte / 365)
                        expected_move_pct = (expected_move / current_price) * 100
                        st.markdown(f'<p style="color:var(--text-light); font-size:1rem; margin-top:1rem;"><b>Expected Move by Expiration:</b> ${expected_move:.2f} ({expected_move_pct:.2f}%)</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color:var(--text-subtle); font-size:0.9rem;">(Based on ATM IV)</p>', unsafe_allow_html=True)
                        
                        # Plotting P/L for the selected option
                        st.markdown('<h4>Profit/Loss Profile at Expiration</h4>', unsafe_allow_html=True)
                        price_range = np.linspace(current_price * 0.9, current_price * 1.1, 100) # +/- 10% range
                        pnl_values = calculate_option_pnl('call', selected_call_row['strike'], selected_call_row['lastPrice'], price_range)
                        
                        pnl_fig = go.Figure()
                        pnl_fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='P/L at Expiration',
                                                        line=dict(color=PRIMARY_ACCENT_COLOR_HEX, width=2)))
                        pnl_fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1) # Breakeven line
                        pnl_fig.update_layout(
                            title=f"Long Call P/L ({selected_call_row['strike']}) at Expiration",
                            xaxis_title="Underlying Price at Expiration",
                            yaxis_title="Profit / Loss ($)",
                            template="plotly_dark",
                            paper_bgcolor=BG_DARK_COLOR_HEX,
                            plot_bgcolor=BG_DARK_COLOR_HEX,
                            font=dict(color=TEXT_LIGHT_COLOR_HEX),
                            hovermode="x unified",
                            height=350,
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        pnl_fig.update_xaxes(showgrid=True, gridcolor=DIVIDER_COLOR_CSS)
                        pnl_fig.update_yaxes(showgrid=True, gridcolor=DIVIDER_COLOR_CSS)
                        st.plotly_chart(pnl_fig, use_container_width=True)

                    else:
                        st.info("Expected Move not calculated (DTE <= 0 or IV <= 0).", icon="ℹ️")
                except Exception as e:
                    st.warning(f"Could not calculate Expected Move/P&L for Call: {e}", icon="⚠️")


            else:
                st.info("Scan found no qualifying calls for this expiration.", icon="🔍")

        with col2:
            st.markdown("<h3>📉 Put Flow</h3>", unsafe_allow_html=True)
            st.markdown("""
                <p style='color:var(--text-subtle); font-size:0.9rem;'>
                    Put options give the holder the right, but not the obligation, to *sell* a stock at a specified strike price.
                    They are generally bought when expecting the stock price to fall.
                </p>
            """, unsafe_allow_html=True)
            top_puts = rank_options(puts, False)
            if not top_puts.empty:
                top_puts['display_name'] = top_puts['contractSymbol'] + ' - $' + top_puts['strike'].astype(str) + ' (IV: ' + (top_puts['impliedVolatility']*100).round(2).astype(str) + '%)'
                put_selection = st.selectbox("Select Put", top_puts["display_name"], key="put_select", help="Select a put option from the ranked list based on liquidity, implied volatility, and proximity to current price.")
                st.dataframe(top_puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'Score']], 
                                 use_container_width=True)
                
                selected_put_row = top_puts[top_puts["display_name"] == put_selection].iloc[0]
                # Pass 'put' as the option_type
                suggestion, reasons, stop, target, rr_ratio, pop = analyze_option(selected_put_row, current_price, dte, 'put')

                st.markdown("<h3>🧠 Analysis</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-box">
                    <p><b>Contract:</b> {selected_put_row['contractSymbol']}</p>
                    <p><b>Strike:</b> ${selected_put_row['strike']:.2f}</p>
                    <p><b>Last:</b> ${selected_put_row['lastPrice']:.2f}</p>
                    <p><b>Spread:</b> ${selected_put_row['bid']:.2f} / ${selected_put_row['ask']:.2f}</p>
                    <p><b>IV:</b> {selected_put_row['impliedVolatility']:.2%}</p>
                    <p><b>Probability of Profit (POP):</b> {(pop*100):.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h4>📊 Trade Setup</h4>", unsafe_allow_html=True)
                rr_display = f"{rr_ratio:.1f}" if not pd.isna(rr_ratio) else "N/A"
                st.markdown(f"""
                - 🎯 Target: ${target:.2f} ({(target/selected_put_row['lastPrice']-1)*100:.0f}%)
                - 🛑 Stop: ${stop:.2f}
                - ⚖️ R/R: {rr_display}
                """)
                
                st.markdown("<h4>🔍 Insights</h4>", unsafe_allow_html=True)
                for r in reasons:
                    st.markdown(f"- {r}")
                
                if "High-Probability" in suggestion:
                    st.success(suggestion, icon="🚀")
                elif "Caution" in suggestion:
                    st.warning(suggestion, icon="⚠️")
                else:
                    st.error(suggestion, icon="🛑")
            else:
                st.info("Scan found no qualifying puts for this expiration.", icon="🔍")

            # Expected Move Calculation & Display for selected option
            try:
                atm_iv = selected_put_row['impliedVolatility']
                
                if dte > 0 and atm_iv > 0:
                    expected_move = current_price * atm_iv * np.sqrt(dte / 365)
                    expected_move_pct = (expected_move / current_price) * 100
                    st.markdown(f'<p style="color:var(--text-light); font-size:1rem; margin-top:1rem;"><b>Expected Move by Expiration:</b> ${expected_move:.2f} ({expected_move_pct:.2f}%)</p>', unsafe_allow_html=True)
                    st.markdown(f'<p style="color:var(--text-subtle); font-size:0.9rem;">(Based on ATM IV)</p>', unsafe_allow_html=True)

                    # Plotting P/L for the selected option
                    st.markdown('<h4>Profit/Loss Profile at Expiration</h4>', unsafe_allow_html=True)
                    price_range = np.linspace(current_price * 0.9, current_price * 1.1, 100) # +/- 10% range
                    pnl_values = calculate_option_pnl('put', selected_put_row['strike'], selected_put_row['lastPrice'], price_range)
                    
                    pnl_fig = go.Figure()
                    pnl_fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='P/L at Expiration',
                                                 line=dict(color=PRIMARY_ACCENT_COLOR_HEX, width=2)))
                    pnl_fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1) # Breakeven line
                    pnl_fig.update_layout(
                        title=f"Long Put P/L ({selected_put_row['strike']}) at Expiration",
                        xaxis_title="Underlying Price at Expiration",
                        yaxis_title="Profit / Loss ($)",
                        template="plotly_dark",
                        paper_bgcolor=BG_DARK_COLOR_HEX,
                        plot_bgcolor=BG_DARK_COLOR_HEX,
                        font=dict(color=TEXT_LIGHT_COLOR_HEX),
                        hovermode="x unified",
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    pnl_fig.update_xaxes(showgrid=True, gridcolor=DIVIDER_COLOR_CSS)
                    pnl_fig.update_yaxes(showgrid=True, gridcolor=DIVIDER_COLOR_CSS)
                    st.plotly_chart(pnl_fig, use_container_width=True)

                else:
                    st.info("Expected Move not calculated (DTE <= 0 or IV <= 0).", icon="ℹ️")
            except Exception as e:
                st.warning(f"Could not calculate Expected Move/P&L for Put: {e}", icon="⚠️")

# --- Tab 4: Dividend Analysis ---
with tab4:
    st.markdown('<div class="section"><h2>💰 Dividend Analysis</h2></div>', unsafe_allow_html=True)

    st.markdown("""
        <p style='color:var(--text-subtle); font-size:0.9rem;'>
            Explore dividend opportunities based on yield, consistency, and value.
            High dividend yield may indicate a good income stock, but consistency and a reasonable valuation are key for long-term investing.
        </p>
    """, unsafe_allow_html=True)

    DIVIDEND_CANDIDATES = [
        "JPM", "KO", "PG", "AAPL", "MSFT", "VZ", "T", "XOM", "CVX", "MMM", 
        "ABBV", "JNJ", "PEP", "DOW", "IBM", "INTC", "CSCO", "DUK", "SO", "NEE"
    ]
    if ticker not in DIVIDEND_CANDIDATES:
        DIVIDEND_CANDIDATES.insert(0, ticker)

    selected_dividend_ticker = st.selectbox(
        "Select a stock for detailed Dividend Analysis",
        DIVIDEND_CANDIDATES,
        key="dividend_ticker_select",
        help="Choose a stock to view its dividend history, yield, and consistency metrics."
    )

    if selected_dividend_ticker:
        @st.cache_data(ttl=3600)
        def get_dividend_data(ticker_symbol):
            try:
                stock = yf.Ticker(ticker_symbol)
                info = stock.info
                dividends = stock.dividends
                
                if not dividends.empty:
                    current_year = datetime.now().year
                    dividends = dividends[dividends.index.year >= current_year - 10]
                
                return info, dividends
            except Exception as e:
                st.warning(f"Could not fetch dividend data for {ticker_symbol}: {e}", icon="⚠️")
                return {}, pd.Series()

        info, dividends_df = get_dividend_data(selected_dividend_ticker)

        if info and not dividends_df.empty:
            st.markdown(f'<div class="section"><h3>Dividend Profile for {selected_dividend_ticker}</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                try:
                    forward_yield = info.get('forwardAnnualDividendYield')
                    if forward_yield is not None:
                            st.markdown(f'<div class="metric-box"><h4>Forward Yield</h4><h3>{forward_yield:.2%}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-box"><h4>Forward Yield</h4><h3>N/A</h3></div>', unsafe_allow_html=True)

                    payout_ratio = info.get('payoutRatio')
                    if payout_ratio is not None:
                        st.markdown(f'<div class="metric-box"><h4>Payout Ratio (Trailing)</h4><h3>{payout_ratio:.2%}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-box"><h4>Payout Ratio (Trailing)</h3><h3>N/A</h3></div>', unsafe_allow_html=True)

                except Exception:
                    st.markdown(f'<div class="metric-box"><h4>Forward Yield</h4><h3>N/A</h3></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box"><h4>Payout Ratio (Trailing)</h4><h3>N/A</h3></div>', unsafe_allow_html=True)

            with col2:
                try:
                    trailing_yield = info.get('trailingAnnualDividendYield')
                    if trailing_yield is not None:
                        st.markdown(f'<div class="metric-box"><h4>Trailing Yield</h4><h3>{trailing_yield:.2%}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-box"><h4>Trailing Yield</h3><h3>N/A</h3></div>', unsafe_allow_html=True)
                    
                    dividend_growth_rate = info.get('fiveYearAvgDividendYield')
                    if dividend_growth_rate is not None:
                        st.markdown(f'<div class="metric-box"><h4>5-Year Avg Yield</h4><h3>{dividend_growth_rate:.2%}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-box"><h4>5-Year Avg Yield</h3><h3>N/A</h3></div>', unsafe_allow_html=True)

                except Exception:
                    # FIX: Corrected the malformed f-string (likely missing 'f' before the string)
                    st.markdown(f'<div class="metric-box"><h4>Trailing Yield</h4><h3>N/A</h3></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box"><h4>5-Year Avg Yield</h4><h3>N/A</h3></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<h4>Dividend History (Last 10 Years)</h4>', unsafe_allow_html=True)
            if not dividends_df.empty:
                annual_dividends = dividends_df.resample('YE').sum().rename('Annual Dividend')
                
                annual_dividends_df = annual_dividends.to_frame()
                annual_dividends_df['Prev Annual Dividend'] = annual_dividends_df['Annual Dividend'].shift(1)
                annual_dividends_df['Growth (%)'] = ((annual_dividends_df['Annual Dividend'] - annual_dividends_df['Prev Annual Dividend']) / annual_dividends_df['Prev Annual Dividend']) * 100
                
                annual_dividends_df.index = annual_dividends_df.index.year
                annual_dividends_df.index.name = 'Year'
                annual_dividends_df = annual_dividends_df[['Annual Dividend', 'Growth (%)']].round(2)
                st.dataframe(annual_dividends_df, use_container_width=True)

                years_paying = len(annual_dividends_df[annual_dividends_df['Annual Dividend'] > 0])
                consistent_message = ""
                if years_paying >= 10 and (annual_dividends_df['Growth (%)'] >= 0).all():
                    consistent_message = "✅ Highly consistent dividend payer with growth over the last 10 years."
                    st.success(consistent_message, icon="⭐")
                elif years_paying >= 5 and (annual_dividends_df['Growth (%)'] >= 0).all():
                    consistent_message = "👍 Consistent dividend payer with recent growth (last 5+ years)."
                    st.info(consistent_message, icon="📈")
                elif years_paying >= 3:
                    consistent_message = f"ℹ️ Paid dividends for {years_paying} consecutive years."
                    st.info(consistent_message, icon="📆")
                else:
                    st.warning(f"🤷 Not enough data or inconsistent payments recently to assess long-term consistency.", icon="⚠️")

            else:
                st.info(f"No dividend history found for {selected_dividend_ticker}.", icon="😔")
        else:
            st.info(f"Could not retrieve a comprehensive dividend profile for {selected_dividend_ticker}. Data might be unavailable or there was an API issue.", icon="🤷")
    
    st.markdown("---")
    st.markdown('<div class="section"><h2>📈 Top Dividend Opportunities</h2></div>', unsafe_allow_html=True)

    @st.cache_data(ttl=4 * 3600)
    def get_top_dividend_stocks(candidate_tickers, min_yield=0.03, min_pe_ratio=20, min_consecutive_years=5):
        results = []
        progress_text = "Analyzing dividend candidates. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, tkr in enumerate(candidate_tickers):
            try:
                stock = yf.Ticker(tkr)
                info = stock.info
                dividends = stock.dividends

                yield_val = info.get('forwardAnnualDividendYield') or info.get('trailingAnnualDividendYield')
                pe_ratio = info.get('trailingPE')
                current_price = info.get('currentPrice')

                consistent_payout = False
                if not dividends.empty:
                    annual_dividends = dividends.resample('YE').sum()
                    consecutive_years_paying = 0
                    last_div_amount = float('inf')
                    for year in sorted(annual_dividends.index.year, reverse=True):
                        if annual_dividends.loc[str(year)].iloc[0] > 0:
                            if annual_dividends.loc[str(year)].iloc[0] >= last_div_amount:
                                consecutive_years_paying += 1
                                last_div_amount = annual_dividends.loc[str(year)].iloc[0]
                            else:
                                break
                        else:
                            break
                    if consecutive_years_paying >= min_consecutive_years:
                        consistent_payout = True

                results.append({
                    "Symbol": tkr,
                    "Current Price": current_price,
                    "Dividend Yield": yield_val,
                    "P/E Ratio": pe_ratio,
                    "Consistent Payout (Years)": consecutive_years_paying,
                    "Is Consistent": consistent_payout,
                })
            except Exception:
                results.append({
                    "Symbol": tkr,
                    "Current Price": None,
                    "Dividend Yield": None,
                    "P/E Ratio": None,
                    "Consistent Payout (Years)": 0,
                    "Is Consistent": False,
                })
            my_bar.progress((i + 1) / len(candidate_tickers), text=f"{progress_text} {i+1}/{len(candidate_tickers)}")
        my_bar.empty()

        df = pd.DataFrame(results)
        df.dropna(subset=["Dividend Yield", "P/E Ratio", "Current Price"], inplace=True)
        return df

    SCANNER_CANDIDATES = list(set(TOP_STOCKS + ["DOW", "IBM", "INTC", "CSCO", "DUK", "SO", "NEE", "MSFT", "PG", "KO", "HD", "XOM", "CVX", "T", "VZ", "PEP", "MCD", "MMM", "ABBV", "JNJ", "PFE"]))
    
    dividend_scan_df = get_top_dividend_stocks(SCANNER_CANDIDATES, min_consecutive_years=5)

    if not dividend_scan_df.empty:
        top_yield_df = dividend_scan_df.sort_values(by="Dividend Yield", ascending=False).head(10).copy()
        top_yield_df.reset_index(drop=True, inplace=True)
        
        st.markdown('<h4>Highest Dividend Yields (Top 10)</h4>', unsafe_allow_html=True)
        st.dataframe(
            top_yield_df[['Symbol', 'Dividend Yield', 'Current Price', 'P/E Ratio', 'Consistent Payout (Years)']]
            .style.format({
                'Dividend Yield': "{:.2%}",
                'Current Price': "${:.2f}",
                'P/E Ratio': "{:.1f}",
                'Consistent Payout (Years)': "{:.0f}"
            }),
            use_container_width=True
        )

        st.markdown("<hr style='border-top: 1px dashed var(--divider-color); margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown('<h4>Consistent Dividend Performers (Yielding & Growth)</h4>', unsafe_allow_html=True)
        consistent_performers_df = dividend_scan_df[
            (dividend_scan_df['Is Consistent'] == True) & 
            (dividend_scan_df['Dividend Yield'] >= 0.02)
        ].sort_values(by="Dividend Yield", ascending=False).head(10).copy()
        consistent_performers_df.reset_index(drop=True, inplace=True)

        if not consistent_performers_df.empty:
            st.dataframe(
                consistent_performers_df[['Symbol', 'Dividend Yield', 'Current Price', 'P/E Ratio', 'Consistent Payout (Years)']]
                .style.format({
                    'Dividend Yield': "{:.2%}",
                    'Current Price': "${:.2f}",
                    'P/E Ratio': "{:.1f}",
                    'Consistent Payout (Years)': "{:.0f}"
                }),
                use_container_width=True
            )
        else:
            st.info("No consistent dividend performers found meeting the criteria.", icon="🔎")

        st.markdown("<hr style='border-top: 1px dashed var(--divider-color); margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown('<h4>Cheap Stocks with Good Dividend Yield (P/E < 15, Yield > 3%)</h4>', unsafe_allow_html=True)
        cheap_dividend_stocks_df = dividend_scan_df[
            (dividend_scan_df['P/E Ratio'] < 15) & 
            (dividend_scan_df['Dividend Yield'] > 0.03)
        ].sort_values(by="Dividend Yield", ascending=False).head(10).copy()
        cheap_dividend_stocks_df.reset_index(drop=True, inplace=True)

        if not cheap_dividend_stocks_df.empty:
            st.dataframe(
                cheap_dividend_stocks_df[['Symbol', 'Dividend Yield', 'Current Price', 'P/E Ratio', 'Consistent Payout (Years)']]
                .style.format({
                    'Dividend Yield': "{:.2%}",
                    'Current Price': "${:.2f}",
                    'P/E Ratio': "{:.1f}",
                    'Consistent Payout (Years)': "{:.0f}"
                }),
                use_container_width=True
            )
        else:
            st.info("No cheap stocks with good dividend yield found meeting the criteria.", icon="🔎")

    else:
        st.warning("Could not retrieve sufficient data for dividend scanning. Please check your internet connection or try again later.", icon="📊")

# --- Searchable Learning Tab (Now tab5) ---
with tab5: # This is now the 5th tab
    st.markdown('<div class="section"><h2>📚 Learning Center</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='color:var(--text-subtle); font-size:0.9rem;'>
            Welcome to the Learning Center! Here you can find essential knowledge about trading, options, and investing.
            Use the navigation below or the search bar to explore different topics.
        </p>
    """, unsafe_allow_html=True)

    # Define all learning content as a dictionary
    LEARNING_CONTENT = {
        "Trading Basics": {
            "summary": "Introduction to buying and selling financial instruments, key terms like assets, bulls vs. bears, long vs. short, liquidity, volatility, and common order types.",
            "content": """
                <h3>Trading Basics: The Foundation</h3>
                <p>Trading involves buying and selling financial instruments with the goal of profiting from price fluctuations. It differs from long-term investing primarily by its shorter time horizons.</p>
                <h4>Key Concepts:</h4>
                <ul>
                    <li><b>Assets:</b> Financial instruments like stocks, bonds, commodities, currencies, cryptocurrencies, etc.</li>
                    <li><b>Bulls vs. Bears:</b> A "bull market" implies rising prices; "bulls" are optimistic traders expecting prices to rise. A "bear market" implies falling prices; "bears" are pessimistic traders expecting prices to fall.</li>
                    <li><b>Long vs. Short:</b> Going "long" means buying an asset expecting its price to rise. Going "short" means selling a borrowed asset expecting its price to fall, then buying it back at a lower price to return it.</li>
                    <li><b>Liquidity:</b> How easily and quickly an asset can be bought or sold without significantly affecting its price. High liquidity is generally preferred for traders as it reduces transaction costs (slippage).</li>
                    <li><b>Volatility:</b> The degree of variation of a trading price series over time. High volatility means prices fluctuate widely, creating more opportunities for profit but also higher risk.</li>
                </ul>
                <h4>Common Order Types:</h4>
                <ul>
                    <li><b>Market Order:</b> An order to immediately buy or sell a security at the best available current price. It guarantees execution but not the price.</li>
                    <li><b>Limit Order:</b> An order to buy or sell a security at a specified price or better. It guarantees the price but not the execution.</li>
                    <li><b>Stop Order:</b> An order to buy or sell once a stock reaches a specified price (the "stop price"). This order becomes a market order when the stop price is hit. Used to limit potential losses or to lock in profits.</li>
                    <li><b>Stop-Limit Order:</b> A combination of a stop order and a limit order. Once the stop price is reached, it becomes a limit order to buy or sell at the specified limit price or better. This offers more price control than a simple stop order.</li>
                </ul>
                <p style='color:var(--text-subtle); font-size:0.85rem;'><i>Disclaimer: Trading involves substantial risk of loss. Only commit capital you can afford to lose.</i></p>
            """
        },
        "Understanding Options": {
            "summary": "Explains call and put options, strike price, expiration date, premium, Greeks (Delta, Gamma, Theta, Vega, Rho), and basic strategies like buying calls/puts, covered calls, and cash-secured puts.",
            "content": """
                <h3>Understanding Options: Leverage & Flexibility</h3>
                <p>Options are financial derivatives that give buyers the right, but not the obligation, to buy or sell an underlying asset at a predetermined price (strike price) before a specified date (expiration date).</p>

                <h4>Types of Options:</h4>
                <ul>
                    <li><b>Call Option:</b> Gives the holder the right to <b>buy</b> the underlying asset at the strike price. Buyers profit if the underlying price rises above the strike price plus premium paid.</li>
                    <li><b>Put Option:</b> Gives the holder the right to <b>sell</b> the underlying asset at the strike price. Buyers profit if the underlying price falls below the strike price minus premium paid.</li>
                </ul>

                <h4>Key Terminology:</h4>
                <ul>
                    <li><b>Strike Price:</b> The fixed price at which the underlying asset can be bought or sold if the option is exercised.</li>
                    <li><b>Expiration Date (DTE - Days To Expiration):</b> The date after which the option contract is no longer valid. Options expire worthless if not exercised or sold by this date.</li>
                    <li><b>Premium:</b> The price paid by the option buyer to the option seller (writer) for the option contract. This is the cost of the option.</li>
                    <li><b>In-the-Money (ITM):</b> An option that has intrinsic value. (Call: underlying price > strike price; Put: underlying price < strike price).</li>
                    <li><b>Out-of-the-Money (OTM):</b> An option that has no intrinsic value. (Call: underlying price < strike price; Put: underlying price > strike price). OTM options only have time value.</li>
                    <li><b>Implied Volatility (IV):</b> A market's forecast of a likely movement in a security's price. Higher IV means higher option premiums due to increased perceived risk/opportunity.</li>
                    <li><b>Greeks (Delta, Gamma, Theta, Vega, Rho):</b> Measures of an option's sensitivity to various factors:
                        <ul>
                            <li><b>Delta (Δ):</b> Sensitivity to changes in the underlying asset's price.</li>
                            <li><b>Gamma (Γ):</b> Rate of change of Delta.</li>
                            <li><b>Theta (Θ):</b> Rate of time decay (loss of value as expiration approaches).</li>
                            <li><b>Vega (ν):</b> Sensitivity to changes in implied volatility.</li>
                            <li><b>Rho (ρ):</b> Sensitivity to changes in interest rates.</li>
                        </ul>
                    </li>
                </ul>
                <h4>Simple Strategies:</h4>
                <ul>
                    <li><b>Buying Calls:</b> A bullish strategy, betting on the underlying price to rise significantly. Max loss is premium paid.</li>
                    <li><b>Buying Puts:</b> A bearish strategy, betting on the underlying price to fall significantly, or used for hedging existing long positions. Max loss is premium paid.</li>
                    <li><b>Covered Call:</b> Selling a call option while simultaneously owning at least 100 shares of the underlying stock. Limits upside potential but generates income (premium).</li>
                    <li><b>Cash-Secured Put:</b> Selling a put option and simultaneously setting aside enough cash to buy the underlying shares if the option is assigned (exercised). Generates income (premium) or allows buying stock at a discount.</li>
                </ul>
                <p style='color:var(--text-subtle); font-size:0.85rem;'><i>Options trading is highly complex and involves significant risk, including the potential for 100% loss of premium paid. Professional guidance is recommended.</i></p>
            """
        },
        "Investing Strategies": {
            "summary": "Approaches to building wealth over time, including value, growth, dividend, and index investing, plus diversification and dollar-cost averaging.",
            "content": """
                <h3>Investing Strategies: Building Wealth Over Time</h3>
                <p>Investing generally involves allocating capital with the expectation of generating a return over a longer period, focusing on growth or income rather than short-term price swings. It's a marathon, not a sprint.</p>
                <h4>Common Approaches:</h4>
                <ul>
                    <li><b>Value Investing:</b> Popularized by Benjamin Graham and Warren Buffett, this strategy involves identifying undervalued companies (e.g., low Price-to-Earnings, high book value, strong balance sheets) whose intrinsic value is higher than their current market price. The investor patiently waits for the market to eventually recognize their true worth.</li>
                    <li><b>Growth Investing:</b> Focuses on companies expected to grow at an above-average rate, even if their current valuation seems high. These are often innovative companies in emerging sectors (e.g., tech startups, biotech firms). Growth investors prioritize revenue and earnings growth over current profitability.</li>
                    <li><b>Dividend Investing:</b> Prioritizing companies that pay regular dividends (a portion of their earnings) to shareholders. The goal is to generate consistent income from your investments. (See our dedicated <a href="#dividend-analysis" style="color:var(--primary-accent);">Dividend Analysis</a> tab for more!)</li>
                    <li><b>Index Investing:</b> Investing in broad market indices (e.g., S&P 500 via ETFs like SPY, or Nasdaq 100 via QQQ) rather than individual stocks. This provides broad diversification, matches overall market performance, and typically has lower fees. It's often recommended for passive investors.</li>
                    <li><b>Dollar-Cost Averaging (DCA):</b> A strategy where an investor invests a fixed amount of money at regular intervals (e.g., $100 every month), regardless of market fluctuations. This reduces the risk of making a large investment at an unfavorable time by averaging out the purchase price over time.</li>
                </ul>
                <h4>Diversification:</h4>
                <p>Spreading investments across various asset classes (stocks, bonds), industries (tech, healthcare), and geographies to reduce overall portfolio risk. The principle is that poor performance in one area might be offset by strong performance in another, reducing the impact of any single negative event. "Don't put all your eggs in one basket."</p>
                <p style='color:var(--text-subtle); font-size:0.85rem;'><i>Investing involves risk, including the potential loss of principal. Past performance is not indicative of future results.</i></p>
            """
        },
        "Dividend Investing": {
            "summary": "Focuses on income generation from stocks, key metrics like dividend yield, payout ratio, and growth rate, and considerations for this strategy.",
            "content": """
                <h3>Dividend Investing: Income Generation</h3>
                <p>Dividend investing focuses on acquiring shares of companies that distribute a portion of their earnings to shareholders in the form of regular dividend payments. This strategy can provide a steady stream of income, especially valuable for retirees or those seeking passive income, alongside potential capital appreciation.</p>
                <h4>Key Metrics for Dividend Stocks:</h4>
                <ul>
                    <li><b>Dividend Yield:</b> Calculated as the annual dividend per share divided by the current share price. A higher yield means more income relative to the amount invested.</li>
                    <li><b>Dividend Payout Ratio:</b> Dividends per share divided by Earnings per Share (EPS). This indicates what percentage of a company's earnings are paid out as dividends. A very high ratio (e.g., consistently above 70-80% for non-REITs/utilities) might indicate an unsustainable dividend, as less is reinvested into the business.</li>
                    <li><b>Dividend Growth Rate (DGR):</b> The average rate at which a company's dividend payments increase over time (e.g., over 3, 5, or 10 years). Consistent dividend growth is a strong indicator of a company's financial health and commitment to returning value to shareholders.</li>
                    <li><b>Dividend Aristocrats/Kings:</b> Specific designations for S&P 500 companies that have increased their dividend for 25+ / 50+ consecutive years, respectively. These are often considered highly reliable dividend payers due to their long track record.</li>
                </ul>
                <h4>Advantages:</h4>
                <ul>
                    <li>Provides a regular income stream, regardless of stock price fluctuations.</li>
                    <li>Companies paying consistent and growing dividends are often financially stable, mature, and well-managed.</li>
                    <li>Dividends can provide a cushion during bear markets, as the income stream remains even if stock prices fall.</li>
                    <li>Reinvesting dividends can significantly compound returns over the long term.</li>
                </ul>
                <h4>Considerations:</h4>
                <ul>
                    <li><b>Sustainability:</b> A high yield alone isn't enough; always check the dividend payout ratio and Free Cash Flow (FCF) to ensure the company can truly afford its payments.</li>
                    <li><b>"Dividend Traps":</b> These are stocks with deceptively high yields due to a falling stock price, often signaling underlying business struggles that could lead to a dividend cut or suspension and significant capital loss.</li>
                    <li><b>Tax Implications:</b> Dividend income is often taxable. Understand the tax rules for qualified vs. non-qualified dividends in your region.</li>
                </ul>
                <p>Explore high-yielding and consistent dividend payers in our dedicated <a href="#dividend-analysis" style="color:var(--primary-accent);">Dividend Analysis</a> tab!</p>
                <p style='color:var(--text-subtle); font-size:0.85rem;'><i>Dividend investing is not without risk. Dividends can be cut or suspended, and stock prices can still decline.</i></p>
            """
        },
        "Risk Management": {
            "summary": "Principles for protecting capital, including position sizing, stop-loss orders, diversification, and understanding risk-reward ratios.",
            "content": """
                <h3>Risk Management: Protecting Your Capital</h3>
                <p>Risk management is paramount in trading and investing. It involves identifying, assessing, and mitigating potential financial risks to protect your capital and ensure long-term sustainability. Without proper risk management, even a high-win-rate strategy can lead to ruin.</p>
                <h4>Core Principles:</h4>
                <ul>
                    <li><b>Capital Preservation:</b> Your primary goal should always be to avoid large, irreversible losses. Protect your principal above all else.</li>
                    <li><b>Position Sizing:</b> Determining how much capital to allocate to any single trade or investment. Never risk more than a small percentage (e.g., 1-2%) of your total trading capital on one trade. This limits the impact of a single losing trade.</li>
                    <li><b>Stop-Loss Orders:</b> An automated order to sell a security if it falls to a predetermined price (the "stop price"). This limits potential losses on a trade.</li>
                    <li><b>Take-Profit Orders:</b> An automated order to sell a security if it reaches a predetermined profit target. This helps lock in gains and prevent emotional decision-making.</li>
                    <li><b>Diversification:</b> Spreading your investments across various asset classes (stocks, bonds), industries (tech, healthcare), and geographies to reduce the impact of a single poor-performing asset or sector.</li>
                    <li><b>Risk-Reward Ratio:</b> The potential profit of a trade relative to its potential loss. For example, a 3:1 ratio means you expect to gain $3 for every $1 you risk. Aim for trades where potential reward significantly outweighs potential risk (e.g., 2:1, 3:1, or higher).</li>
                    <li><b>Emotional Discipline:</b> Adhering strictly to your predefined trading plan and risk management rules, avoiding impulsive decisions driven by fear, greed, or frustration.</li>
                </ul>
                <h4>Why Risk Management Matters:</h4>
                <p>Even with a high win rate, poor risk management can lead to catastrophic losses. A few large, uncontrolled losses can wipe out many small gains, leading to irreversible damage to your portfolio. Conversely, even with a moderate win rate, strict risk management can lead to consistent long-term profitability by controlling drawdowns and protecting capital.</p>
                <p style='color:var(--text-subtle); font-size:0.85rem;'><i>Effective risk management is the cornerstone of successful trading and investing. It cannot be overstated.</i></p>
            """
        }
    }

    # Search bar for learning content
    search_query = st.text_input("Search our Learning Center...", key="learning_search_input", help="Type keywords to find relevant topics (e.g., 'options greeks', 'stop loss', 'dividend yield').")
    
    st.markdown("---")

    found_results = []
    if search_query:
        search_query_lower = search_query.lower()
        for title, item_data in LEARNING_CONTENT.items():
            # Search in title, summary, and content
            if search_query_lower in title.lower() or \
               search_query_lower in item_data["summary"].lower() or \
               search_query_lower in item_data["content"].lower():
                found_results.append({"title": title, "summary": item_data["summary"], "content": item_data["content"]})
        
        if found_results:
            st.markdown(f"<h4>Found {len(found_results)} result(s) for '{search_query}':</h4>", unsafe_allow_html=True)
            for result in found_results:
                st.markdown(f"""
                <div class="news-card">
                    <h4>{result['title']}</h4>
                    <p>{result['summary']}</p>
                    <p style="font-size:0.9em; color:var(--text-subtle);">Click the Learning Center radio button above to read the full article.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No results found for '{search_query}'. Please try a different query or browse the topics above.", icon="🔎")
    else:
        # Display clickable radio buttons to navigate topics
        learning_page_selection = st.radio(
            "Or select a topic to read:",
            list(LEARNING_CONTENT.keys()),
            key="learning_module_display_select",
            horizontal=True
        )
        if learning_page_selection:
            st.markdown(LEARNING_CONTENT.get(learning_page_selection, "Content not found."), unsafe_allow_html=True)


# --- Watchlist Tab (New tab6) ---
# Initialize watchlist in session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["NVDA", "AAPL", "MSFT"] # Default watchlist items

with tab6: # This is now the 6th tab
    st.markdown('<div class="section"><h2>⭐ My Watchlist</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='color:var(--text-subtle); font-size:0.9rem;'>
            Add or remove stocks to your personal watchlist to track their current performance and Market Insights at a glance.
        </p>
    """, unsafe_allow_html=True)

    # Input for adding/removing tickers
    col_add, col_remove = st.columns([0.7, 0.3])
    with col_add:
        new_ticker_to_add = st.text_input("Add Ticker to Watchlist", key="add_ticker_input").upper()
    with col_remove:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
        if st.button("Add Stock", key="add_stock_button"):
            if new_ticker_to_add and new_ticker_to_add not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker_to_add)
                st.success(f"Added {new_ticker_to_add} to watchlist!")
            elif new_ticker_to_add in st.session_state.watchlist:
                st.warning(f"{new_ticker_to_add} is already in your watchlist.")
            else:
                st.info("Please enter a ticker symbol to add.")
    
    # Remove ticker functionality
    if st.session_state.watchlist:
        tickers_to_remove = st.multiselect("Remove Ticker(s) from Watchlist", options=st.session_state.watchlist, key="remove_ticker_multiselect")
        if st.button("Remove Selected", key="remove_selected_button"):
            if tickers_to_remove:
                for tkr_to_remove in tickers_to_remove:
                    st.session_state.watchlist.remove(tkr_to_remove)
                st.success(f"Removed {', '.join(tickers_to_remove)} from watchlist.")
            else:
                st.info("Please select tickers to remove.")
    else:
        st.info("Your watchlist is currently empty. Add some stocks!", icon="💡")

    st.markdown("---")
    st.markdown('<h3>Watchlist Insights</h3>', unsafe_allow_html=True)

    if st.session_state.watchlist:
        # Re-using get_overview_data for watchlist insights
        # Pass global_sentiment_analyzer directly for caching, as it's a fixed resource.
        # This function doesn't rely on OpenAI explicitly, only NLTK sentiment.
        watchlist_overview_df = get_overview_data(st.session_state.watchlist, API_KEY, global_sentiment_analyzer)
        
        if not watchlist_overview_df.empty:
            watchlist_overview_df = watchlist_overview_df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
            st.dataframe(
                watchlist_overview_df[['Symbol', 'Score', 'Price Up Today', 'High Volume', 'Above VWAP', 'Healthy RSI (30-70)', 'MACD Bullish Cross', 'Positive News Sentiment']],
                use_container_width=True
            )
        else:
            st.warning("Could not retrieve live data ", icon="📊")
    else:
        st.info("Add stocks to your watchlist to see their insights here!", icon="💡")