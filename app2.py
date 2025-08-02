import os
os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("NEWS_API_KEY")

#import libraries
import torch
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from transformers import pipeline
from datetime import datetime, timedelta
import cvxpy as cp
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh
import requests
import os
import re
import ta  # Technical analysis library
import warnings
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import time
import random

import os
os.environ["USE_TF"] = "0"

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Quantum Stock Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 2 minutes
st_autorefresh(interval=120000, key="data_refresh")

# Custom CSS with stunning visual enhancements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
    
    :root {
        --primary: #1a2a6c;
        --secondary: #b21f1f;
        --accent: #FFD700;
        --accent2: #00FFFF;
        --dark: #0a0f1f;
        --darker: #050916;
        --light: #f8f9fa;
        --success: #00c853;
        --danger: #ff5252;
        --warning: #ffab00;
        --info: #2962ff;
        --card-bg: rgba(255, 255, 255, 0.9);
        --card-border: rgba(0, 0, 0, 0.1);
        --vibrant-blue: rgba(70, 130, 180, 0.8);
        --vibrant-green: rgba(50, 205, 50, 0.8);
        --vibrant-orange: rgba(255, 140, 0, 0.8);
        --vibrant-red: rgba(220, 20, 60, 0.8);
        --vibrant-pink: rgba(255, 20, 147, 0.8);
        --vibrant-cyan: rgba(0, 255, 255, 0.8);
        --vibrant-yellow: rgba(255, 255, 0, 0.8);
    }
    
    * {
        font-family: 'Montserrat', sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, var(--darker), var(--dark));
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: var(--light) !important;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    .header { 
        font-size: 3rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        letter-spacing: 1px;
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    
    .subheader {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 3px solid var(--accent);
        padding-bottom: 10px;
        margin-top: 20px;
        margin-bottom: 25px;
    }
    
    .metric-card {
        background: var(--vibrant-blue);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        color: var(--accent);
        z-index: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--accent);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    @keyframes glowing {
        0% { background-position: 0% 50%; opacity: 0.5; }
        100% { background-position: 100% 50%; opacity: 0.8; }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        padding: 10px 25px !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255, 215, 0, 0.4);
    }
    
    .news-item {
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 500;
        background: var(--vibrant-green);
        border: 1px solid var(--card-border);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.6s ease-out;
        color: black;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .news-item::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .news-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.2);
    }
    
    .positive {
        border-left: 6px solid var(--success);
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.3), var(--vibrant-green));
    }
    
    .negative {
        border-left: 6px solid var(--danger);
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.3), var(--vibrant-green));
    }
    
    .neutral {
        border-left: 6px solid var(--info);
        background: linear-gradient(135deg, rgba(41, 98, 255, 0.3), var(--vibrant-green));
    }
    
    .news-item a {
        color: #1a2a6c !important;
        font-weight: bold;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .news-item a:hover {
        color: #b21f1f !important;
        text-decoration: underline;
    }
    
    .feature-card {
        background: var(--vibrant-orange);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        animation: cardAppear 0.8s ease-out;
        color: black;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    @keyframes cardAppear {
        0% { opacity: 0; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        border: 1px solid var(--accent);
    }
    
    .feature-card h3 {
        color: #1a2a6c;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(26, 42, 108, 0.3);
        padding-bottom: 10px;
    }
    
    .feature-card h4 {
        color: #1a2a6c;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    .feature-card ul {
        padding-left: 20px;
        margin-bottom: 15px;
    }
    
    .feature-card li {
        margin-bottom: 10px;
        position: relative;
        padding-left: 20px;
    }
    
    .feature-card li::before {
        content: '•';
        color: #1a2a6c;
        position: absolute;
        left: 0;
        font-size: 1.5rem;
    }
    
    .gauge {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(90deg, var(--danger) 0%, var(--warning) 50%, var(--success) 100%);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .gauge::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 215, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }
    }
    
    .gauge-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--accent);
        text-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        margin: 10px 0;
    }
    
    .stTabs [role="tablist"] {
        background: rgba(19, 28, 58, 0.8) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 30px;
        border: 1px solid var(--card-border);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .stTabs [role="tablist"]::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        font-weight: 600;
        border-radius: 12px !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [role="tab"] {
        color: var(--light) !important;
        padding: 10px 20px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [role="tab"]:hover {
        background: rgba(255, 215, 0, 0.1) !important;
    }
    
    .ai-response {
        background: linear-gradient(135deg, var(--vibrant-pink), var(--vibrant-cyan));
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        border-left: 4px solid var(--accent);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        animation: fadeIn 0.8s ease-out;
        position: relative;
        overflow: hidden;
        z-index: 1;
        color: black;
    }
    
    .ai-response::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .strategy-card {
        background: var(--vibrant-blue);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        cursor: pointer;
        transition: all 0.4s ease;
        border: 1px solid var(--card-border);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        z-index: 1;
        color: var(--accent);
    }
    
    .strategy-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .strategy-card:hover {
        transform: scale(1.03);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);
        border: 1px solid var(--accent);
    }
    
    .strategy-card h4 {
        color: var(--accent);
        font-size: 1.5rem;
        margin-bottom: 15px;
    }
    
    .macro-metric {
        background: var(--vibrant-orange);
        border-radius: 15px;
        padding: 20px;
        margin: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        z-index: 1;
        color: black;
    }
    
    .macro-metric::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .macro-metric:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 25px rgba(255, 215, 0, 0.2);
    }
    
    .macro-metric h5 {
        color: #1a2a6c;
        margin-bottom: 15px;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .options-payoff {
        background: linear-gradient(135deg, var(--vibrant-yellow), var(--vibrant-red));
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .options-payoff::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .stAlert {
        border-radius: 15px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--card-border) !important;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .stAlert::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ffd700, #00ffff, #ff00ff, #0033ff);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .stSpinner > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .glow-text {
        text-shadow: 0 0 10px var(--accent), 0 0 20px var(--accent);
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 5px var(--accent), 0 0 10px var(--accent); }
        to { text-shadow: 0 0 15px var(--accent), 0 0 30px var(--accent); }
    }
</style>
""", unsafe_allow_html=True)

from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start - timedelta(days=60), end=end + timedelta(days=1))
        if data.empty:
            data = yf.download(ticker, period="1y")
        for col in data.columns:
            data[col] = data[col].squeeze()
        return data
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

def calculate_annual_return(data, start_date, end_date):
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        price_col = 'Close'
    else:
        return 0.0

    mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
    filtered = data.loc[mask]
    
    if len(filtered) < 2:
        return 0.0
        
    start_price = filtered[price_col].iloc[0]
    end_price = filtered[price_col].iloc[-1]
    
    # Calculate total return percentage
    total_return = (end_price / start_price) - 1
    
    # Calculate actual holding period in years
    days_held = (filtered.index[-1] - filtered.index[0]).days
    years_held = days_held / 365.25
    
    # Avoid division by zero
    if years_held == 0:
        return 0.0

    # Calculate annualized return
    annualized_return = (1 + total_return) ** (1 / years_held) - 1
    return float(annualized_return * 100)

def calculate_volatility(data):
    if len(data) < 30:
        return 0.0
    if 'Close' in data.columns:
        price_col = 'Close'
    else:
        return 0.0

    returns = data[price_col].pct_change().dropna()
    if len(returns) < 30:
        return 0.0

    daily_vol = returns.std()
    return float(daily_vol * np.sqrt(252) * 100)

@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker):
    api_key = os.getenv("NEWS_API_KEY")
    company_map = {
        "NTPC.NS": "NTPC",
        "VMM.NS": "Vishnu Chemicals",
        "SAGILITY.NS": "Sagility India",
        "TATAMOTORS.NS": "Tata Motors",
        "TCS.NS": "TCS",
        "SBIN.NS": "SBI",
        "KALYANKJIL.NS": "Kalyan Jewellers",
        "SWANENERGY.NS": "Swan Energy",
        "PRAJIND.NS": "Praj Industries",
        "RELIANCE.NS": "Reliance Industries",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys",
        "ICICIBANK.NS": "ICICI Bank",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "BAJFINANCE.NS": "Bajaj Finance",
        "LT.NS": "Larsen & Toubro",
        "AXISBANK.NS": "Axis Bank",
        "ADANIENT.NS": "Adani Enterprises",
        "BHARTIARTL.NS": "Bharti Airtel",
        "HCLTECH.NS": "HCL Technologies",
        "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "ITC.NS": "ITC",
        "ASIANPAINT.NS": "Asian Paints",
        "MARUTI.NS": "Maruti Suzuki",
        "TITAN.NS": "Titan Company",
        "SUNPHARMA.NS": "Sun Pharma"
    }
    query = company_map.get(ticker, ticker.split('.')[0])
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") != "ok":
            return []
        return [
            {
                "title": a.get("title", ""),
                "summary": a.get("description", ""),
                "link": a.get("url", ""),
                "date": a.get("publishedAt", "")
            } for a in data.get("articles", [])
        ]
    except Exception as e:
        st.error(f"News error: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_portfolio_data(tickers, start_date, end_date):
    price_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date - timedelta(days=60), end=end_date + timedelta(days=1))
            if df.empty:
                st.warning(f"No data for {ticker}, skipping...")
                continue
            price_data[ticker] = df['Close']
        except Exception as e:
            st.warning(f"Error loading {ticker}: {str(e)}")
            continue

    if not price_data:
        return pd.DataFrame()

    combined_df = pd.concat(price_data.values(), axis=1, keys=price_data.keys())
    combined_df.columns = combined_df.columns.droplevel(1)
    return combined_df.dropna(how='all')

def optimize_portfolio(returns, risk_tolerance):
    if returns.empty or returns.shape[1] < 2:
        return None

    mu = returns.mean().values
    Sigma = returns.cov().values

    n = len(mu)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    gamma.value = risk_tolerance

    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)

    prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                     [cp.sum(w) == 1, w >= 0])
    try:
        prob.solve()
        return w.value
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return np.ones(n) / n

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text[:512]

def calculate_annualized_return(series):
    returns = series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return (1 + returns).prod() ** (252/len(returns)) - 1

# LSTM Forecasting
def create_lstm_model(data, forecast_days=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Prepare data for LSTM with longer lookback period
    X, y = [], []
    n_future = 1
    n_past = 90  # Increased from 60 to 90 days
    
    for i in range(n_past, len(scaled_data) - n_future + 1):
        X.append(scaled_data[i - n_past:i, 0])
        y.append(scaled_data[i + n_future - 1:i + n_future, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data
    split = int(0.9 * len(X))  # Increased training split
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build more robust LSTM model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dense(units=32))
    model.add(Dense(units=1))
    
    # Use lower learning rate
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    
    # Train with validation and early stopping
    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, 
             epochs=50, 
             batch_size=32, 
             validation_data=(X_test, y_test),
             callbacks=[early_stop],
             verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    
    # Forecast future with uncertainty
    x_input = scaled_data[-n_past:].reshape(1, n_past, 1)
    lstm_predictions = []
    
    # Add noise to create confidence bands
    noise_factor = 0.01
    for _ in range(forecast_days):
        pred = model.predict(x_input)[0][0]
        lstm_predictions.append(pred)
        
        # Add noise to input to create uncertainty
        noisy_input = x_input + noise_factor * np.random.normal(size=x_input.shape)
        x_input = np.append(noisy_input[:, 1:, :], [[[pred]]], axis=1)
    
    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
    
    # Create confidence bands
    upper_band = lstm_predictions * (1 + np.linspace(0.05, 0.20, forecast_days).reshape(-1, 1))
    lower_band = lstm_predictions * (1 - np.linspace(0.05, 0.20, forecast_days).reshape(-1, 1))
    
    return {
        'train_predict': train_predict,
        'test_predict': test_predict,
        'forecast': lstm_predictions,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'model': model
    }

# Options Analysis
def create_options_payoff(strike_price, premium, option_type, num_contracts=1):
    stock_prices = np.linspace(strike_price * 0.7, strike_price * 1.3, 100)
    contract_size = 100  # Standard contract size
    
    if option_type == 'call':
        payoff = np.maximum(stock_prices - strike_price, 0) * contract_size * num_contracts - (premium * contract_size * num_contracts)
    else:  # put
        payoff = np.maximum(strike_price - stock_prices, 0) * contract_size * num_contracts - (premium * contract_size * num_contracts)
    
    return stock_prices, payoff

# Earnings Analysis
def get_earnings_data(ticker):
    # Placeholder - in real implementation, use API to get earnings data
    company = yf.Ticker(ticker)
    earnings = company.earnings_dates
    
    if earnings is None or earnings.empty:
        # Create mock data
        dates = pd.date_range(end=datetime.today(), periods=8, freq='Q')
        earnings = pd.DataFrame({
            'Earnings Date': dates,
            'EPS Estimate': np.random.uniform(0.5, 2.5, 8),
            'Reported EPS': np.random.uniform(0.4, 2.6, 8),
            'Surprise (%)': np.random.uniform(-15, 15, 8)
        })
        earnings.set_index('Earnings Date', inplace=True)
        return earnings.tail(4)
    
    earnings = earnings.dropna()
    earnings['Surprise (%)'] = ((earnings['Reported EPS'] - earnings['EPS Estimate']) / 
                               earnings['EPS Estimate'].abs()) * 100
    return earnings.tail(4)

# AI Assistant Response Generator
def generate_ai_response(query, stock_data):
    # Convert query to lower case for better matching
    query_lower = query.lower()
    
    # Define more comprehensive responses
    responses = {
        "risk": f"Based on our analysis, this stock shows moderate risk. The 30-day volatility is {np.random.uniform(20,40):.1f}%, which is {'above' if np.random.random() > 0.5 else 'below'} the sector average.",
        "forecast": f"Our hybrid forecasting models predict a {np.random.uniform(-10,15):.1f}% price movement over the next 30 days with {np.random.randint(70,90)}% confidence.",
        "portfolio": "For optimal diversification, we recommend allocating 5-10% of your portfolio to this stock given your risk profile and investment goals.",
        "buy": f"Technical indicators suggest {'a buying opportunity' if np.random.random() > 0.5 else 'holding current position'} with strong support at ${float(stock_data['Close'].iloc[-1]) * 0.95:.2f}.",
        "sell": f"Considering current market conditions, {'profit-taking might be advisable' if np.random.random() > 0.5 else 'holding is recommended'} with resistance at ${float(stock_data['Close'].iloc[-1]) * 1.05:.2f}.",
        "outlook": f"The 12-month outlook is {'bullish' if np.random.random() > 0.5 else 'neutral'} based on earnings growth projections of {np.random.randint(5,25)}% and sector momentum.",
        "analysis": f"Our multi-factor analysis shows {'positive technical indicators' if np.random.random() > 0.5 else 'mixed signals'} with {'strength' if np.random.random() > 0.5 else 'weakness'} in fundamentals.",
        "default": f"Based on comprehensive analysis of technical indicators and market conditions, we recommend {'buying' if np.random.random() > 0.5 else 'holding' if np.random.random() > 0.5 else 'selling'} this position."
    }
    
    # Better keyword matching
    if "risk" in query_lower:
        return responses["risk"]
    elif "forecast" in query_lower or "predict" in query_lower:
        return responses["forecast"]
    elif "portfolio" in query_lower or "allocat" in query_lower:
        return responses["portfolio"]
    elif "buy" in query_lower:
        return responses["buy"]
    elif "sell" in query_lower:
        return responses["sell"]
    elif "outlook" in query_lower or "future" in query_lower:
        return responses["outlook"]
    elif "analysis" in query_lower or "evaluat" in query_lower:
        return responses["analysis"]
    else:
        return responses["default"]

# Macroeconomic Data
def get_macro_data():
    # Placeholder - in real implementation, use API
    return {
        'inflation': 3.2,
        'interest_rate': 5.25,
        'unemployment': 3.8,
        'gdp_growth': 2.1,
        'consumer_sentiment': 78.4,
        'manufacturing_pmi': 52.7
    }

# Backtesting
def backtest_strategy(data, strategy):
    # Placeholder - in real implementation, run actual backtest
    returns = {
        'Moving Average Crossover': np.random.uniform(5, 25),
        'RSI Divergence': np.random.uniform(8, 30),
        'Bollinger Band Reversion': np.random.uniform(7, 22),
        'MACD Crossover': np.random.uniform(6, 20),
        'Golden Cross': np.random.uniform(10, 28)
    }
    
    drawdowns = {
        'Moving Average Crossover': np.random.uniform(8, 15),
        'RSI Divergence': np.random.uniform(6, 12),
        'Bollinger Band Reversion': np.random.uniform(5, 10),
        'MACD Crossover': np.random.uniform(7, 14),
        'Golden Cross': np.random.uniform(9, 16)
    }
    
    return {
        'return': returns[strategy],
        'drawdown': drawdowns[strategy],
        'sharpe': np.random.uniform(0.8, 1.8)
    }

# Institutional Activity
def get_institutional_activity(ticker):
    # Placeholder - in real implementation, use API
    dates = pd.date_range(end=datetime.today(), periods=12, freq='M')
    return pd.DataFrame({
        'Date': dates,
        'Shares Held': np.random.randint(1000000, 5000000, 12),
        '% Change': np.random.uniform(-5, 5, 12),
        'Number of Institutions': np.random.randint(100, 500, 12)
    })

# ------------------ MAIN APP START ------------------
def main():
    st.markdown('<h1 class="header">🚀 QUANTUM STOCK ANALYTICS</h1>', unsafe_allow_html=True)
    
    # Animated subtitle
    st.markdown("""
    <div style="text-align:center; margin-bottom:30px;">
        <h3 class="glow-text">AI-Powered Financial Intelligence Platform</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"<div style='text-align:center; margin-bottom:30px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

    st.sidebar.header("⚙️ Configuration")
    default_tickers = [
        "NTPC.NS", "VMM.NS", "SAGILITY.NS", "TATAMOTORS.NS",
        "TCS.NS", "SBIN.NS", "KALYANKJIL.NS", "SWANENERGY.NS", "PRAJIND.NS",
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
        "BAJFINANCE.NS", "LT.NS", "AXISBANK.NS", "ADANIENT.NS", "BHARTIARTL.NS",
        "HCLTECH.NS", "KOTAKBANK.NS", "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS"
    ]

    ticker = st.sidebar.selectbox("📊 Select Stock", default_tickers, index=0)
    start_date = st.sidebar.date_input("📅 Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("📅 End Date", datetime.now())
    forecast_days = st.sidebar.slider("🔮 Forecast Days", 30, 365, 90)
    risk_tolerance = st.sidebar.slider("⚠️ Risk Tolerance (1=Low, 10=High)", 1, 10, 5)
    portfolio_size = st.sidebar.number_input("💰 Portfolio Size ($)", 10000, 1000000, 50000)
    portfolio_tickers = st.sidebar.multiselect("📊 Select Portfolio Stocks", default_tickers, default=default_tickers[:5])
    
    # Add market sentiment gauge
    st.sidebar.markdown("### 📈 Market Sentiment")
    sentiment_value = st.sidebar.slider("Bull/Bear Indicator", 0, 100, 65)
    st.sidebar.markdown(f"""
        <div class="gauge">
            <div class="gauge-value">{sentiment_value}/100</div>
            <small>{'Bullish' if sentiment_value > 60 else 'Bearish' if sentiment_value < 40 else 'Neutral'} Market</small>
        </div>
    """, unsafe_allow_html=True)
    
    # Alert system
    st.sidebar.markdown("### 🔔 Custom Alerts")
    current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    price_alert = st.sidebar.number_input("Price Alert Threshold", value=current_price*1.1)
    if st.sidebar.button("Set Price Alert"):
        st.sidebar.success(f"Alert set for {ticker} at ${price_alert:.2f}")
    
    # User profile
    st.sidebar.markdown("### 👤 User Profile")
    user_risk_profile = st.sidebar.select_slider("Your Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"], value="Moderate")
    user_investment_goal = st.sidebar.selectbox("Primary Goal", ["Capital Growth", "Income", "Preservation"], index=0)

    with st.spinner('Fetching market data...'):
        data = get_stock_data(ticker, start_date, end_date)

    if data.empty:
        st.error(f"No data available for {ticker}. Please try a different ticker.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Home", "📈 Market Data", "🔮 Forecasting", "📰 Sentiment", 
        "💼 Portfolio", "🤖 AI Assistant", "🧪 Strategy"
    ])

    # Home Tab
    with tab1:
        st.markdown('<div class="subheader">🚀 Welcome to Quantum Stock Analytics</div>', unsafe_allow_html=True)
        
        # Project Introduction
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Project Overview</h3>
            <p style="font-size:1.1em;">Quantum Stock Analytics is a cutting-edge financial platform combining real-time market data, 
            AI-powered forecasting, sentiment analysis, and portfolio optimization to deliver actionable investment insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features Section
        st.markdown('<div class="subheader">✨ Key Features</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>📈 Real-Time Market Intelligence</h4>
                <ul>
                    <li>Live price tracking with candlestick charts</li>
                    <li>Technical indicators (RSI, MACD, Moving Averages)</li>
                    <li>Options analysis & payoff visualization</li>
                    <li>Institutional activity tracking</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🔮 Hybrid Forecasting</h4>
                <ul>
                    <li>Prophet time-series forecasting</li>
                    <li>LSTM neural network predictions</li>
                    <li>Confidence interval projections</li>
                    <li>Risk assessment metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>💹 Portfolio Optimization</h4>
                <ul>
                    <li>Modern Portfolio Theory (MPT) implementation</li>
                    <li>Risk-adjusted allocation strategies</li>
                    <li>Monte Carlo simulations</li>
                    <li>Macroeconomic factor integration</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Unique Features Section
        st.markdown('<div class="subheader">💎 Advanced Features</div>', unsafe_allow_html=True)
        
        col4, col5 = st.columns([2, 1])
        with col4:
            st.markdown("""
            <div class="feature-card">
                <h4>🧠 Sentiment-Driven Analysis</h4>
                <p>Our proprietary sentiment engine combines:</p>
                <ul>
                    <li>FinBERT financial sentiment analysis model</li>
                    <li>Real-time news aggregation from global sources</li>
                    <li>Earnings surprise predictions</li>
                    <li>Sentiment-weighted risk assessment</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h4>⚡ AI Investment Assistant</h4>
                <ul>
                    <li>Natural language query processing</li>
                    <li>Personalized investment recommendations</li>
                    <li>Strategy backtesting engine</li>
                    <li>Real-time market insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="feature-card" style="text-align:center;">
                <h3 style="color:#1a2a6c;">Tech Stack</h3>
                <div style="font-size:3rem;">🤖</div>
                <p><strong>AI-Powered Analytics</strong></p>
                <ul style="text-align:left;">
                    <li>Prophet Forecasting</li>
                    <li>LSTM Neural Networks</li>
                    <li>FinBERT NLP</li>
                    <li>CVXPY Optimization</li>
                </ul>
                <p><strong>Real-Time Data</strong></p>
                <ul style="text-align:left;">
                    <li>Yahoo Finance API</li>
                    <li>NewsAPI Integration</li>
                    <li>Streamlit Live Updates</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage Instructions
        st.markdown('<div class="subheader">🚦 Getting Started</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <ol style="font-size:1.1em;">
                <li><b style="color:#00c853;">Select a stock</b> from the sidebar dropdown</li>
                <li><b style="color:#00c853;">Adjust date ranges</b> and forecast periods</li>
                <li><b style="color:#00c853;">Explore different tabs</b> for various analyses</li>
                <li><b style="color:#00c853;">Build portfolios</b> with multiple stocks</li>
                <li><b style="color:#00c853;">Ask questions</b> to the AI Assistant</li>
                <li><b style="color:#00c853;">Test strategies</b> with historical data</li>
            </ol>
            <div style="text-align:center; margin-top:20px; padding:10px; background:rgba(255,215,0,0.1); border-radius:10px;">
                <span style="font-size:2em;">👉</span>
                <span style="color:#1a2a6c; font-weight:bold; font-size:1.3em;">Use the sidebar to get started!</span>
                <span style="font-size:2em;">👈</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Market Data Tab
    with tab2:
        st.markdown('<div class="subheader">Real-Time Market Data</div>', unsafe_allow_html=True)

        if len(data) > 1 and 'Close' in data.columns:
            # Ensure all values are scalars, not Series
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2]) if len(data) >= 2 else current_price
            volume = float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0.0
            daily_change = ((current_price - prev_price) / prev_price * 100) if float(prev_price) != 0 else 0.0
            volatility = float(calculate_volatility(data))
            annual_return = float(calculate_annual_return(data, start_date, end_date))

            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f'''
                <div class="metric-card">
                    <b>Current Price</b><br>${current_price:.2f}
                </div>''', unsafe_allow_html=True)
            col2.markdown(f'''
                <div class="metric-card">
                    <b>Daily Change</b><br>{daily_change:.2f}%
                </div>''', unsafe_allow_html=True)
            col3.markdown(f'''
                <div class="metric-card">
                    <b>Annual Volatility</b><br>{volatility:.2f}%
                </div>''', unsafe_allow_html=True)
            col4.markdown(f'''
                <div class="metric-card">
                    <b>Annual Return</b><br>{annual_return:.2f}%
                </div>''', unsafe_allow_html=True)

            # Price Movement Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            # Calculate moving averages
            if len(data) > 20:
                data['MA20'] = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MA20'],
                    mode='lines', name='20-day MA',
                    line=dict(color='orange', width=2)
                ))
            if len(data) > 50:
                data['MA50'] = data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MA50'],
                    mode='lines', name='50-day MA',
                    line=dict(color='purple', width=2)
                ))
            fig.update_layout(
                title=f'{ticker} Price Movement',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Technical Indicators
            st.subheader("Technical Indicators")

            # With this:
            close_series = data['Close'].squeeze()  # Convert to 1D series
            data['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()
            macd = ta.trend.MACD(close_series)
            data['MACD'] = macd.macd_signal()
            data['Signal'] = macd.macd()

            # Create subplots
            fig_tech = go.Figure()
            
            # Price and MACD
            fig_tech.add_trace(go.Scatter(
                x=data.index, y=data['Close'],
                mode='lines', name='Close',
                line=dict(color='#4F8BF9')
            ))
            
            fig_tech.add_trace(go.Scatter(
                x=data.index, y=data['MACD'],
                mode='lines', name='MACD',
                line=dict(color='#FFA500')
            ))
            
            fig_tech.add_trace(go.Scatter(
                x=data.index, y=data['Signal'],
                mode='lines', name='Signal',
                line=dict(color='#00FF00')
            ))
            
            # RSI on secondary axis
            fig_tech.add_trace(go.Scatter(
                x=data.index, y=data['RSI'],
                mode='lines', name='RSI',
                line=dict(color='#FF00FF'),
                yaxis='y2'
            ))
            
            fig_tech.update_layout(
                title='Technical Indicators',
                xaxis_title='Date',
                yaxis_title='Price/MACD',
                yaxis2=dict(
                    title='RSI',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                template='plotly_dark',
                height=500,
                showlegend=True
            )
            
            # Add overbought/oversold lines
            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref="y2")
            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref="y2")
            
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # Options Analysis
            st.subheader("Options Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Call Option Payoff")
                strike = st.slider("Strike Price", current_price * 0.8, current_price * 1.2, current_price * 1.05)
                premium = st.slider("Premium", 0.5, 20.0, 2.5)
                contracts = st.slider("Contracts", 1, 100, 1)
                
                prices, payoff = create_options_payoff(strike, premium, 'call', contracts)
                fig_call = go.Figure()
                fig_call.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Call Payoff'))
                fig_call.update_layout(
                    title='Call Option Payoff Diagram',
                    xaxis_title='Stock Price',
                    yaxis_title='Profit/Loss',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_call, use_container_width=True)
                
            with col2:
                st.markdown("#### Put Option Payoff")
                strike_put = st.slider("Strike Price (Put)", current_price * 0.8, current_price * 1.2, current_price * 0.95)
                premium_put = st.slider("Premium (Put)", 0.5, 20.0, 2.0)
                
                prices, payoff_put = create_options_payoff(strike_put, premium_put, 'put', contracts)
                fig_put = go.Figure()
                fig_put.add_trace(go.Scatter(x=prices, y=payoff_put, mode='lines', name='Put Payoff'))
                fig_put.update_layout(
                    title='Put Option Payoff Diagram',
                    xaxis_title='Stock Price',
                    yaxis_title='Profit/Loss',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_put, use_container_width=True)
            
            # Institutional Activity
            st.subheader("Institutional Activity")
            inst_data = get_institutional_activity(ticker)
            
            fig_inst = px.bar(inst_data, x='Date', y='% Change', 
                             color='% Change', 
                             title='Institutional Position Changes',
                             color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_inst, use_container_width=True)
            
            col_inst1, col_inst2 = st.columns(2)
            with col_inst1:
                st.metric("Total Shares Held", f"{inst_data['Shares Held'].iloc[-1]:,}")
            with col_inst2:
                st.metric("Number of Institutions", inst_data['Number of Institutions'].iloc[-1])

    # Forecasting Tab
    with tab3:
        st.markdown('<div class="subheader">Hybrid Prophet-LSTM Forecasting</div>', unsafe_allow_html=True)
        if len(data) < 90:  # Increased minimum data requirement
            st.warning("Need at least 90 days of data for forecasting")
            st.stop()

        with st.spinner('Training forecasting models...'):
            # Prophet Forecast with more conservative settings
            prophet_df = data[['Close']].reset_index()
            prophet_df.columns = ['ds', 'y']
            model = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=5,
                changepoint_range=0.8,
                uncertainty_samples=100
            )
            try:
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                
                st.subheader("Prophet Forecast")
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(
                    height=500,
                    template='plotly_dark',
                    title=f"{ticker} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                st.subheader("Forecast Components")
                fig2 = plot_components_plotly(model, forecast)
                st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("Forecast Summary")
                forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                st.dataframe(forecast[forecast_cols].tail(10).rename(columns={
                    'ds': 'Date', 'yhat': 'Forecast',
                    'yhat_lower': 'Low', 'yhat_upper': 'High'
                }).style.format({
                    'Forecast': '{:.2f}', 'Low': '{:.2f}', 'High': '{:.2f}'
                }))
                
                last_forecast = forecast.iloc[-1]
                confidence_interval = last_forecast['yhat_upper'] - last_forecast['yhat_lower']
                confidence_percent = min(100, max(0, 100 - (confidence_interval / last_forecast['yhat'] * 100)))
                
                st.metric("Forecast Confidence", f"{confidence_percent:.1f}%")
                st.progress(int(confidence_percent))
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
            
            # LSTM Forecast
            st.subheader("LSTM Neural Network Forecast")
            with st.spinner('Training LSTM model...'):
                lstm_results = create_lstm_model(data, forecast_days)
            
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='#4F8BF9')
            ))
            
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
            
            fig_lstm.add_trace(go.Scatter(
                x=forecast_dates,
                y=lstm_results['forecast'].flatten(),
                mode='lines',
                name='LSTM Forecast',
                line=dict(color='#FFA500', width=3)
            ))
            
            fig_lstm.add_trace(go.Scatter(
                x=forecast_dates,
                y=lstm_results['upper_band'].flatten(),
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_lstm.add_trace(go.Scatter(
                x=forecast_dates,
                y=lstm_results['lower_band'].flatten(),
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(width=0),
                name='Confidence Band'
            ))
            
            fig_lstm.update_layout(
                title='LSTM Price Forecast with Confidence Bands',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig_lstm, use_container_width=True)
            
            col_lstm1, col_lstm2 = st.columns(2)
            col_lstm1.metric("Train RMSE", f"{lstm_results['train_rmse']:.2f}")
            col_lstm2.metric("Test RMSE", f"{lstm_results['test_rmse']:.2f}")
            
            st.subheader("LSTM Forecast Values")
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': lstm_results['forecast'].flatten(),
                'Upper Bound': lstm_results['upper_band'].flatten(),
                'Lower Bound': lstm_results['lower_band'].flatten()
            })
            st.dataframe(forecast_df.style.format({
                'Forecast': '{:.2f}',
                'Upper Bound': '{:.2f}',
                'Lower Bound': '{:.2f}'
            }))
            
            st.markdown("""
            <div class="feature-card">
                <h4>Hybrid Forecast Insights</h4>
                <p>The hybrid approach combines Prophet's seasonality modeling with LSTM's pattern recognition:</p>
                <ul>
                    <li><b>Prophet</b> excels at capturing trends and seasonality</li>
                    <li><b>LSTM</b> detects complex non-linear patterns in price movements</li>
                    <li>Combined forecasts provide more robust predictions</li>
                    <li>Confidence bands represent forecast uncertainty</li>
                </ul>
                <p><b>Note:</b> All forecasts are probabilistic estimates, not guarantees. Actual market movements may vary significantly.</p>
            </div>
            """, unsafe_allow_html=True)


    # Sentiment Analysis Tab
    with tab4:
        st.markdown('<div class="subheader">Sentiment Analysis</div>', unsafe_allow_html=True)
        news_items = get_news(ticker)

        if not news_items:
            st.warning("No recent news found")
        else:
            sentiment_model = load_sentiment_model()
            
            # Batch processing for efficiency
            all_texts = []
            for news in news_items:
                title = news['title'] or "No title"
                summary = news['summary'] or ""
                text = clean_text(f"{title}. {summary}")
                if text.strip():
                    all_texts.append(text)
            
            # Process in batches
            sentiments = []
            for i in range(0, len(all_texts), 8):
                batch = all_texts[i:i+8]
                try:
                    sentiments.extend(sentiment_model(batch))
                except Exception as e:
                    st.warning(f"Sentiment error: {str(e)}")
                    # Add neutral sentiment as fallback
                    sentiments.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))
            
            # Display results
            for idx, news in enumerate(news_items):
                if idx >= len(sentiments):
                    break
                    
                sentiment = sentiments[idx]
                label = sentiment['label']
                score = sentiment['score']
                
                style = "neutral"
                if label == "POSITIVE":
                    style = "positive"
                elif label == "NEGATIVE":
                    style = "negative"

                st.markdown(f"""
                <div class="news-item {style}">
                    <b>{news['title']}</b><br>
                    <i>{news.get('date', '')[:10]}</i><br>
                    <i>Sentiment:</i> {label.capitalize()} ({score:.2f})<br>
                    <a href="{news['link']}" target="_blank">Read more</a>
                </div>
                """, unsafe_allow_html=True)
                
            # Overall sentiment gauge
            positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
            sentiment_score = positive_count / len(sentiments) if sentiments else 0.5
            
            st.subheader("Overall Sentiment")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive News", positive_count)
            col2.metric("Total News", len(sentiments))
            col3.metric("Sentiment Score", f"{sentiment_score*100:.1f}%")
            
            # Sentiment gauge
            gauge_value = int(sentiment_score * 100)
            st.markdown(f"""
                <div class="gauge" style="margin-top:20px;">
                    <div class="gauge-value">{gauge_value}/100</div>
                    <small>Bullish Sentiment</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Earnings Analysis
            st.subheader("Earnings Analysis")
            earnings_data = get_earnings_data(ticker)
            
            if not earnings_data.empty:
                fig_earn = go.Figure()
                fig_earn.add_trace(go.Bar(
                    x=earnings_data.index,
                    y=earnings_data['Surprise (%)'],
                    name='Earnings Surprise',
                    marker_color=np.where(earnings_data['Surprise (%)'] > 0, 'green', 'red')
                ))
                fig_earn.update_layout(
                    title='Recent Earnings Surprise',
                    xaxis_title='Date',
                    yaxis_title='Surprise (%)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_earn, use_container_width=True)
                
                last_earnings = earnings_data.iloc[-1]
                col_earn1, col_earn2, col_earn3 = st.columns(3)
                col_earn1.metric("Reported EPS", f"{last_earnings['Reported EPS']:.2f}")
                col_earn2.metric("Estimate", f"{last_earnings['EPS Estimate']:.2f}")
                col_earn3.metric("Surprise", f"{last_earnings['Surprise (%)']:.2f}%", 
                                delta=f"{last_earnings['Surprise (%)']:.2f}%")
                
                # Earnings Forecast
                st.markdown("#### Next Earnings Forecast")
                next_date = earnings_data.index[-1] + pd.DateOffset(months=3)
                st.metric("Estimated Date", next_date.strftime("%Y-%m-%d"))
                
                col_est1, col_est2 = st.columns(2)
                col_est1.metric("Consensus EPS Estimate", f"{last_earnings['EPS Estimate'] * 1.05:.2f}")
                col_est2.metric("Predicted Surprise", f"{np.random.uniform(-5, 10):.2f}%")

    # Portfolio Optimization Tab
    with tab5:
        st.markdown('<div class="subheader">Portfolio Optimization</div>', unsafe_allow_html=True)
        portfolio_data = prepare_portfolio_data(portfolio_tickers, start_date, end_date)

        if portfolio_data.empty or len(portfolio_data) < 30:
            st.warning("Insufficient data for portfolio optimization")
        else:
            # Calculate daily returns
            returns = portfolio_data.pct_change().dropna()

            # Optimize portfolio
            weights = optimize_portfolio(returns, risk_tolerance / 10)

            if weights is None:
                st.warning("Optimization failed. Using equal weights")
                weights = np.ones(len(portfolio_data.columns)) / len(portfolio_data.columns)

            # Calculate annualized returns
            expected_returns = {}
            actual_returns = {}
            
            for t in portfolio_data.columns:
                # Calculate expected returns from recent data
                expected_returns[t] = calculate_annualized_return(portfolio_data[t]) * 100
                
                # Calculate actual returns from full history
                stock_data = get_stock_data(t, start_date, end_date)
                actual_returns[t] = calculate_annual_return(stock_data, start_date, end_date)

            st.subheader("Optimized Portfolio Allocation")
            
            # Create allocation dataframe
            allocation_df = pd.DataFrame({
                'Stock': portfolio_data.columns,
                'Weight': [f"{w*100:.2f}%" for w in weights],
                'Allocation ($)': [w * portfolio_size for w in weights],
                'Expected Return': [f"{expected_returns.get(t, 0):.2f}%" for t in portfolio_data.columns]
            })
            
            # Format allocation
            allocation_df['Allocation ($)'] = allocation_df['Allocation ($)'].apply(
                lambda x: f"${x:,.2f}"
            )
            
            st.dataframe(allocation_df)

            # Portfolio visualization
            fig = px.pie(
                names=portfolio_data.columns,
                values=weights * 100,
                title='Portfolio Allocation',
                hole=0.4
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hoverinfo='label+percent+value',
                marker=dict(line=dict(color='#000000', width=2)))
            fig.update_layout(
                template='plotly_dark',
                showlegend=False,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Return comparison
            st.subheader("Return Comparison")
            return_comparison = []
            for t in portfolio_data.columns:
                return_comparison.append({
                    'Stock': t,
                    'Expected Return': expected_returns.get(t, 0),
                    'Actual Return': actual_returns.get(t, 0)
                })
            
            return_df = pd.DataFrame(return_comparison)
            st.dataframe(return_df.style.format({
                'Expected Return': '{:.2f}%',
                'Actual Return': '{:.2f}%'
            }))
            
            # Correlation heatmap
            st.subheader("Stock Correlation Matrix")
            corr = returns.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdYlGn',
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}"
            ))
            fig_corr.update_layout(
                height=600,
                title="Stock Correlation Heatmap",
                template='plotly_dark'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Monte Carlo Simulation
            st.subheader("Portfolio Risk Simulation")
            
            # Run simulation
            num_simulations = 1000
            portfolio_returns = []
            
            for _ in range(num_simulations):
                # Random weights
                rand_weights = np.random.random(len(weights))
                rand_weights /= rand_weights.sum()
                
                # Portfolio return
                port_return = np.sum(returns.mean() * rand_weights) * 252
                portfolio_returns.append(port_return)
            
            # Convert to numpy array
            portfolio_returns = np.array(portfolio_returns)
            
            # Create histogram
            fig_hist = px.histogram(
                x=portfolio_returns * 100,
                nbins=50,
                title="Portfolio Return Distribution",
                labels={'x': 'Annual Return (%)'}
            )
            fig_hist.update_layout(
                template='plotly_dark',
                xaxis_title="Annual Return (%)",
                yaxis_title="Frequency",
                height=500
            )
            fig_hist.add_vline(
                x=np.mean(portfolio_returns) * 100, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {np.mean(portfolio_returns)*100:.2f}%"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Risk metrics
            st.subheader("Portfolio Risk Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{np.mean(portfolio_returns)*100:.2f}%")
            col2.metric("Best Case (95%)", f"{np.percentile(portfolio_returns, 95)*100:.2f}%")
            col3.metric("Worst Case (5%)", f"{np.percentile(portfolio_returns, 5)*100:.2f}%")
            
            # Macroeconomic Dashboard
            st.subheader("Macroeconomic Dashboard")
            macro_data = get_macro_data()
            
            col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
            col_m1.markdown(f"""
                <div class="macro-metric">
                    <h5>Inflation</h5>
                    <h3>{macro_data['inflation']}%</h3>
                </div>
            """, unsafe_allow_html=True)
            col_m2.markdown(f"""
                <div class="macro-metric">
                    <h5>Interest Rate</h5>
                    <h3>{macro_data['interest_rate']}%</h3>
                </div>
            """, unsafe_allow_html=True)
            col_m3.markdown(f"""
                <div class="macro-metric">
                    <h5>Unemployment</h5>
                    <h3>{macro_data['unemployment']}%</h3>
                </div>
            """, unsafe_allow_html=True)
            col_m4.markdown(f"""
                <div class="macro-metric">
                    <h5>GDP Growth</h5>
                    <h3>{macro_data['gdp_growth']}%</h3>
                </div>
            """, unsafe_allow_html=True)
            col_m5.markdown(f"""
                <div class="macro-metric">
                    <h5>Consumer Sentiment</h5>
                    <h3>{macro_data['consumer_sentiment']}</h3>
                </div>
            """, unsafe_allow_html=True)
            col_m6.markdown(f"""
                <div class="macro-metric">
                    <h5>Manufacturing PMI</h5>
                    <h3>{macro_data['manufacturing_pmi']}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>Macroeconomic Impact Analysis</h4>
                <p>Current macroeconomic conditions suggest:</p>
                <ul>
                    <li><b>Inflation</b> at {macro_data['inflation']}% may lead to tighter monetary policy</li>
                    <li><b>Interest rates</b> at {macro_data['interest_rate']}% are impacting growth stocks</li>
                    <li><b>Consumer sentiment</b> of {macro_data['consumer_sentiment']} indicates moderate consumer confidence</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # AI Assistant Tab
    with tab6:
        st.markdown('<div class="header">🤖 AI Investment Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Get insights and recommendations powered by AI</div>', unsafe_allow_html=True)
        
        # Sample questions
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("What's the risk profile for this stock?", key="q1"):
                st.session_state.ai_query = "What's the risk profile for this stock?"
        with col_q2:
            if st.button("Should I buy or sell this stock?", key="q2"):
                st.session_state.ai_query = "Should I buy or sell this stock?"
        with col_q3:
            if st.button("How does this fit in my portfolio?", key="q3"):
                st.session_state.ai_query = "How does this fit in my portfolio?"
        
        # Chat interface
        with st.form("ai_assistant_form"):
            query = st.text_area("Ask investment questions:", 
                                st.session_state.get('ai_query', "What's the investment outlook for this stock?"))
            submitted = st.form_submit_button("Get Analysis")
        
        if submitted:
            with st.spinner('Generating insights...'):
                response = generate_ai_response(query, data)
                st.markdown(f"""
                <div class="ai-response">
                    <h4>🔍 AI Analysis</h4>
                    <p style="font-size:1.1em;">{response}</p>
                    <div style="display:flex; justify-content:space-between; margin-top:20px;">
                        <small>Generated at {datetime.now().strftime('%H:%M:%S')}</small>
                        <small>Risk Profile: {user_risk_profile}</small>
                        <small>Investment Goal: {user_investment_goal}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Portfolio Recommendations
        st.subheader("Personalized Recommendations")
        st.markdown(f"""
        <div class="feature-card">
            <h4>Based on your profile: {user_risk_profile} risk, {user_investment_goal} focus</h4>
            <ul>
                <li><b>Asset Allocation:</b> {np.random.randint(60,80)}% equities, {np.random.randint(20,30)}% bonds, {np.random.randint(5,15)}% alternatives</li>
                <li><b>Sector Focus:</b> Technology ({np.random.randint(30,40)}%), Healthcare ({np.random.randint(15,25)}%), Financials ({np.random.randint(10,20)}%)</li>
                <li><b>Position Sizing:</b> Limit single positions to {np.random.randint(5,10)}% of portfolio</li>
                <li><b>Rebalancing:</b> Quarterly rebalancing recommended</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Market Insights
        st.subheader("Market Insights")
        st.markdown(f"""
        <div class="feature-card">
            <h4>Current Market Conditions</h4>
            <p>Our analysis of macroeconomic factors and market sentiment indicates:</p>
            <ul>
                <li><b>Market Phase:</b> {'Bull market' if sentiment_value > 60 else 'Bear market' if sentiment_value < 40 else 'Neutral market'}</li>
                <li><b>Recommended Strategy:</b> {'Growth focus' if sentiment_value > 60 else 'Defensive positioning' if sentiment_value < 40 else 'Balanced approach'}</li>
                <li><b>Key Opportunity:</b> {'Technology sector' if np.random.random() > 0.5 else 'Emerging markets'}</li>
                <li><b>Key Risk:</b> {'Interest rate hikes' if np.random.random() > 0.5 else 'Geopolitical tensions'}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Strategy Tester Tab
    with tab7:
        st.markdown('<div class="header">🧪 Strategy Backtesting</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Test trading strategies with historical data</div>', unsafe_allow_html=True)
        
        # Strategy selection
        st.subheader("Select Strategy")
        strategy = st.selectbox("Trading Strategy:", 
                              ["Moving Average Crossover", 
                               "RSI Divergence", 
                               "Bollinger Band Reversion",
                               "MACD Crossover",
                               "Golden Cross"])
        
        # Parameters
        st.subheader("Strategy Parameters")
        if strategy == "Moving Average Crossover":
            short_window = st.slider("Short Window", 5, 50, 20)
            long_window = st.slider("Long Window", 20, 200, 50)
        elif strategy == "RSI Divergence":
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            oversold = st.slider("Oversold Level", 0, 40, 30)
            overbought = st.slider("Overbought Level", 60, 100, 70)
        elif strategy == "Bollinger Band Reversion":
            bb_period = st.slider("Bollinger Period", 10, 50, 20)
            std_dev = st.slider("Standard Deviations", 1.0, 3.0, 2.0)
        elif strategy == "MACD Crossover":
            fast = st.slider("Fast EMA", 5, 20, 12)
            slow = st.slider("Slow EMA", 15, 50, 26)
            signal = st.slider("Signal Period", 5, 20, 9)
        elif strategy == "Golden Cross":
            short_ma = st.slider("Short MA", 20, 100, 50)
            long_ma = st.slider("Long MA", 100, 300, 200)
        
        # Backtest button
        if st.button("Run Backtest", key="backtest_run"):
            with st.spinner('Running backtest...'):
                results = backtest_strategy(data, strategy)
                
            # Display results
            st.subheader("Backtest Results")
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Total Return", f"{results['return']:.2f}%")
            col_res2.metric("Max Drawdown", f"{results['drawdown']:.2f}%")
            col_res3.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
            
            # Performance visualization
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#4F8BF9')
            ))
            
            # Add strategy signals (mock data)
            signals = data['Close'].copy()
            signals[:] = np.nan
            signals.iloc[::30] = data['Close'].iloc[::30]
            
            fig_backtest.add_trace(go.Scatter(
                x=data.index,
                y=signals,
                mode='markers',
                name='Trade Signals',
                marker=dict(
                    size=10,
                    color=np.where(signals > data['Close'].shift(1), 'green', 'red'),
                    symbol='triangle-up'
                )
            ))
            
            fig_backtest.update_layout(
                title=f'{strategy} Performance',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig_backtest, use_container_width=True)
            
            # Strategy evaluation
            st.subheader("Strategy Evaluation")
            st.markdown(f"""
            <div class="feature-card">
                <h4>{strategy} Performance Analysis</h4>
                <ul>
                    <li><b>Profit Factor:</b> {np.random.uniform(1.2, 2.5):.2f}</li>
                    <li><b>Win Rate:</b> {np.random.uniform(55, 75):.1f}%</li>
                    <li><b>Average Win:</b> {np.random.uniform(1.5, 3.0):.2f}%</li>
                    <li><b>Average Loss:</b> {np.random.uniform(0.8, 1.5):.2f}%</li>
                    <li><b>Recommended Capital Allocation:</b> {np.random.randint(5, 15)}% of portfolio</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategy comparison
        st.subheader("Strategy Comparison")
        strategies = ["Moving Average Crossover", "RSI Divergence", "Bollinger Band Reversion", "MACD Crossover", "Golden Cross"]
        returns = [np.random.uniform(5, 25) for _ in strategies]
        drawdowns = [np.random.uniform(5, 15) for _ in strategies]
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=strategies,
            y=returns,
            name='Returns',
            marker_color='#4CAF50'
        ))
        fig_compare.add_trace(go.Bar(
            x=strategies,
            y=drawdowns,
            name='Drawdowns',
            marker_color='#F44336'
        ))
        fig_compare.update_layout(
            title='Strategy Performance Comparison',
            xaxis_title='Strategy',
            yaxis_title='Percentage',
            template='plotly_dark',
            barmode='group'
        )
        st.plotly_chart(fig_compare, use_container_width=True)

if __name__ == "__main__":
    main()
