import streamlit as st

# Claude-inspired color palette
PRIMARY_COLOR = "#CB9B6B"  # Warm copper/brown
ACCENT_COLOR = "#8B7355"   # Dark copper
SUCCESS_COLOR = "#10B981"   # Green
DANGER_COLOR = "#EF4444"    # Red
WARNING_COLOR = "#F59E0B"   # Amber
INFO_COLOR = "#6366F1"      # Indigo

# Neutrals (Claude-style)
BG_PRIMARY = "#FDFCFA"      # Warm off-white
BG_SECONDARY = "#F5F3F0"    # Light beige
BORDER_COLOR = "#E5E1DB"    # Soft border
TEXT_PRIMARY = "#1F1F1F"    # Almost black
TEXT_SECONDARY = "#6B6B6B"  # Gray
TEXT_MUTED = "#999999"      # Light gray

def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Reset */
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Main App Background */
        .stApp {{
            background-color: {BG_PRIMARY};
        }}
        
        /* Main Container */
        .main .block-container {{
            padding: 2rem 3rem;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_PRIMARY};
            font-weight: 600;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: 1.75rem;
            margin-bottom: 1rem;
        }}
        
        /* Paragraphs */
        p {{
            color: {TEXT_SECONDARY};
            line-height: 1.6;
        }}
        
        /* Tabs - Claude Style */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            background-color: transparent;
            border-bottom: 1px solid {BORDER_COLOR};
            padding: 0;
            margin-bottom: 2rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.75rem 1.5rem !important;
            color: {TEXT_MUTED} !important;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            color: {TEXT_SECONDARY} !important;
            background: transparent !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: transparent !important;
            color: {PRIMARY_COLOR} !important;
            border-bottom: 2px solid {PRIMARY_COLOR} !important;
            font-weight: 600;
        }}
        
        /* Cards */
        .card {{
            background: white;
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: box-shadow 0.2s ease;
        }}
        
        .card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: white;
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s ease;
        }}
        
        .metric-card:hover {{
            border-color: {PRIMARY_COLOR};
            box-shadow: 0 2px 8px rgba(203, 155, 107, 0.15);
        }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: {TEXT_MUTED};
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            line-height: 1;
        }}
        
        /* Buttons */
        .stButton > button {{
            background: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.65rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .stButton > button:hover {{
            background: {ACCENT_COLOR};
            box-shadow: 0 4px 12px rgba(203, 155, 107, 0.3);
            transform: translateY(-1px);
        }}
        
        /* Selectbox & Inputs */
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput input {{
            background-color: white !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 8px;
            color: {TEXT_PRIMARY} !important;
            font-size: 0.95rem;
        }}
        
        .stSelectbox div[data-baseweb="select"] > div:focus-visible,
        .stTextInput input:focus {{
            border-color: {PRIMARY_COLOR} !important;
            box-shadow: 0 0 0 3px rgba(203, 155, 107, 0.1) !important;
        }}
        
        /* Dropdown Options */
        div[data-baseweb="popover"] {{
            background: white !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
        }}
        
        div[data-baseweb="popover"] [role="option"] {{
            color: {TEXT_PRIMARY} !important;
            padding: 0.75rem 1rem !important;
        }}
        
        div[data-baseweb="popover"] [role="option"]:hover {{
            background-color: {BG_SECONDARY} !important;
        }}
        
        div[data-baseweb="popover"] [aria-selected="true"] {{
            background-color: rgba(203, 155, 107, 0.1) !important;
            color: {PRIMARY_COLOR} !important;
            font-weight: 500 !important;
        }}
        
        /* Sidebar */
        .stSidebar {{
            background-color: white;
            border-right: 1px solid {BORDER_COLOR};
        }}
        
        .stSidebar .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {BG_SECONDARY} !important;
        }}
        
        /* DataFrames */
        .stDataFrame {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .stDataFrame thead th {{
            background-color: {BG_SECONDARY} !important;
            color: {TEXT_PRIMARY} !important;
            font-weight: 600 !important;
            border-bottom: 1px solid {BORDER_COLOR} !important;
            padding: 0.875rem !important;
        }}
        
        .stDataFrame tbody td {{
            color: {TEXT_PRIMARY} !important;
            border-bottom: 1px solid {BORDER_COLOR} !important;
            padding: 0.75rem !important;
        }}
        
        .stDataFrame tbody tr:hover {{
            background-color: {BG_SECONDARY} !important;
        }}
        
        /* Alert Boxes */
        .stAlert {{
            border-radius: 8px;
            border: 1px solid {BORDER_COLOR};
            padding: 1rem;
        }}
        
        /* Info boxes */
        .element-container .stMarkdown {{
            color: {TEXT_SECONDARY};
        }}
        
        /* Progress Bars */
        .stProgress > div > div {{
            background-color: {PRIMARY_COLOR};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: white;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            color: {TEXT_PRIMARY};
        }}
        
        /* Success/Error/Warning colors */
        .positive {{
            color: {SUCCESS_COLOR};
        }}
        
        .negative {{
            color: {DANGER_COLOR};
        }}
        
        .neutral {{
            color: {TEXT_MUTED};
        }}
        
        /* Hide Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {BG_SECONDARY};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {PRIMARY_COLOR};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {ACCENT_COLOR};
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

# Export colors for use in components
PRIMARY_ACCENT_COLOR_HEX = PRIMARY_COLOR
SECONDARY_ACCENT_COLOR_HEX = ACCENT_COLOR
TEXT_LIGHT_COLOR_HEX = TEXT_PRIMARY
TEXT_SUBTLE_COLOR_HEX = TEXT_SECONDARY
BG_DARK_COLOR_HEX = BG_PRIMARY
BG_DARKER_COLOR_HEX = BG_SECONDARY
PANEL_BG_COLOR_CSS = "white"
BORDER_COLOR_CSS = BORDER_COLOR
DIVIDER_COLOR_CSS = BORDER_COLOR
SUCCESS_COLOR_HEX = SUCCESS_COLOR
DANGER_COLOR_HEX = DANGER_COLOR
WARNING_COLOR_HEX = WARNING_COLOR
INFO_COLOR_HEX = INFO_COLOR
BORDER_COLOR_FOR_MPL_PLOTLY = BORDER_COLOR
DIVIDER_COLOR_FOR_MPL_PLOTLY = BORDER_COLOR