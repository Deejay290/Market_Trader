# Market_Trader
Market Application that helps track all aspects of the market and predict bullish and bearish markets.

Install libraries 
   ``` Terminal
   pip install streamlit yfinance pandas numpy matplotlib plotly nltk pytz scikit-learn scipy requests
   ```

Download NLTK data
   ```Terminal
   python3 -c "import nltk; nltk.download('vader_lexicon')"
   ```
Set up API key(optional)
   
   Create `.streamlit/secrets.toml`:
   ```toml
   API_KEY = "your_finnhub_api_key_here"
   ```
   Get a free key from [Finnhub](https://finnhub.io/register)

Launch the viewer
   ```Terminal
   streamlit run TradeBot_V1.py
   ```

Open in browser
   
   Will automatically open at `http://localhost:8501`
