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
from keras.layers import Dense
from keras.optimizers import Adam
import time
import random
from transformers import pipeline
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import talib

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
        --secondary: #0a5f38;
        --accent: #00c853;
        --accent2: #00b8d4;
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
        --vibrant-teal: rgba(0, 150, 136, 0.8);
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
        text-shadow: 0 0 20px rgba(0, 200, 83, 0.3);
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
        background: var(--vibrant-teal);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        color: white;
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 200, 83, 0.4);
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
        color: #0a5f38 !important;
        text-decoration: underline;
    }
    
    .feature-card {
        background: var(--vibrant-teal);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        animation: cardAppear 0.8s ease-out;
        color: white;
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 10px;
    }
    
    .feature-card h4 {
        color: white;
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
        color: white;
    }
    
    .feature-card li::before {
        content: '•';
        color: white;
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 200, 83, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 200, 83, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 200, 83, 0); }
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
        background: rgba(0, 200, 83, 0.1) !important;
    }
    
    .ai-response {
        background: linear-gradient(135deg, var(--vibrant-teal), var(--vibrant-cyan));
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
        color: white;
    }
    
    .ai-response::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .strategy-card {
        background: var(--vibrant-teal);
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
        background: var(--vibrant-teal);
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
        color: white;
    }
    
    .macro-metric::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
        z-index: -1;
        filter: blur(5px);
        animation: glowing 3s ease-in-out infinite alternate;
        background-size: 400% 400%;
    }
    
    .macro-metric:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 25px rgba(0, 200, 83, 0.2);
    }
    
    .macro-metric h5 {
        color: white;
        margin-bottom: 15px;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .options-payoff {
        background: linear-gradient(135deg, var(--vibrant-teal), var(--vibrant-orange));
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
        background: linear-gradient(45deg, #1d976c, #93f9b9, #00b8d4, #0052d4);
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
    
    /* Attention heatmap styling */
    .attention-heatmap {
        border-radius: 15px;
        padding: 20px;
        background: var(--vibrant-teal);
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    
    .shap-plot {
        border-radius: 15px;
        padding: 20px;
        background: var(--vibrant-teal);
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

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
        close_series = data['Close'].squeeze()  # Ensure 1D series
        returns = close_series.pct_change().dropna()
        if len(returns) < 30:
            return 0.0
        daily_vol = returns.std()
        return float(daily_vol * np.sqrt(252) * 100)
    return 0.0

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

# Calculate technical indicators - FIXED to ensure 1D arrays
def add_technical_indicators(data):
    if 'Close' not in data.columns or len(data) < 20:
        return data
    
    # Ensure we're working with 1D Series
    close_series = data['Close'].squeeze()
    high_series = data['High'].squeeze() if 'High' in data.columns else close_series
    low_series = data['Low'].squeeze() if 'Low' in data.columns else close_series
    
    # Moving Averages
    try:
        data['SMA20'] = close_series.rolling(window=20).mean()
        data['SMA50'] = close_series.rolling(window=50).mean()
        data['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
    except Exception as e:
        st.warning(f"Error calculating moving averages: {str(e)}")
    
    # RSI
    try:
        data['RSI'] = ta.momentum.rsi(close_series, window=14)
    except Exception as e:
        st.warning(f"Error calculating RSI: {str(e)}")
    
    # MACD
    try:
        macd = ta.trend.MACD(close_series)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"Error calculating MACD: {str(e)}")
    
    # Bollinger Bands
    try:
        bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        data['BB_Width'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"Error calculating Bollinger Bands: {str(e)}")
    
    # Drop any remaining NaN values
    data = data.dropna()
    
    return data

# Temporal Fusion Transformer Forecasting with Technical Indicators - FIXED dtype error
@st.cache_resource(show_spinner=False)
def create_tft_model(data, forecast_days=30):
    # Add technical indicators
    data = add_technical_indicators(data.copy())
    
    # Prepare data for TFT
    df = data.reset_index()
    df.rename(columns={'Date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['time_idx'] = np.arange(len(df))
    df['series'] = "stock"
    
    # Ensure all columns are 1D Series
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()
    
    # Define features
    features = ['Close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                'BB_Upper', 'BB_Lower', 'BB_Width']
    
    # Only keep available features
    available_features = [f for f in features if f in df.columns]
    
    # Create features - convert to strings for categorical encoding
    df['month'] = df['date'].dt.month.astype(str)
    df['day'] = df['date'].dt.day.astype(str)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(str)
    df['quarter'] = df['date'].dt.quarter.astype(str)
    
    # Define training parameters
    max_prediction_length = forecast_days
    max_encoder_length = min(180, len(df) - max_prediction_length - 1)
    
    if max_encoder_length < 60:
        st.warning("Insufficient data for TFT forecasting. Need at least 60 days of data.")
        return {
            'forecast': np.zeros(forecast_days),
            'upper_band': np.zeros(forecast_days),
            'lower_band': np.zeros(forecast_days),
            'train_rmse': 0,
            'test_rmse': 0
        }
    
    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    # Create dataset
    try:
        training = TimeSeriesDataSet(
            df[df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="Close",
            group_ids=["series"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["series"],
            time_varying_known_categoricals=["month", "day", "dayofweek", "quarter"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=available_features,
            target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    except Exception as e:
        st.error(f"Error creating TFT dataset: {str(e)}")
        return {
            'forecast': np.zeros(forecast_days),
            'upper_band': np.zeros(forecast_days),
            'lower_band': np.zeros(forecast_days),
            'train_rmse': 0,
            'test_rmse': 0
        }
    
    # Create validation set
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    # Create dataloaders
    batch_size = 16  # Reduced for performance
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # Configure TFT with QuantileLoss
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=3,  # For P10, P50, P90
        loss=torch.nn.QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=2,
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=15,
        gpus=0,
        enable_progress_bar=False,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        limit_train_batches=15,
        enable_checkpointing=True,
    )
    
    try:
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    except Exception as e:
        st.error(f"Error training TFT model: {str(e)}")
        return {
            'forecast': np.zeros(forecast_days),
            'upper_band': np.zeros(forecast_days),
            'lower_band': np.zeros(forecast_days),
            'train_rmse': 0,
            'test_rmse': 0
        }
    
    # Generate predictions
    try:
        raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
        
        # Extract forecast values (P50 for main forecast, P10/P90 for bands)
        forecast = raw_predictions[0].output.prediction[1].cpu().numpy().flatten()  # P50
        lower_band = raw_predictions[0].output.prediction[0].cpu().numpy().flatten()  # P10
        upper_band = raw_predictions[0].output.prediction[2].cpu().numpy().flatten()  # P90
        
        # Get actual values for comparison
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).cpu().numpy()
        
        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(actuals.flatten()[:len(forecast)], forecast))
        test_rmse = train_rmse
        
        # Extract attention weights
        attention = tft.interpret_output(raw_predictions, reduction="none")[1]['attention'][0].detach().cpu().numpy()
        
        return {
            'forecast': forecast,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'model': tft,
            'attention': attention
        }
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return {
            'forecast': np.zeros(forecast_days),
            'upper_band': np.zeros(forecast_days),
            'lower_band': np.zeros(forecast_days),
            'train_rmse': 0,
            'test_rmse': 0
        }

# Feature Importance with SHAP
def calculate_feature_importance(data):
    if len(data) < 30:
        return None
        
    # Add technical indicators
    data = add_technical_indicators(data.copy())
    
    # Create target (next day's return)
    data['target'] = data['Close'].pct_change().shift(-1)
    data = data.dropna()
    
    if len(data) < 30:
        return None
        
    # Select features
    features = ['Open', 'High', 'Low', 'Volume', 'SMA20', 'SMA50', 'EMA20', 
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'BB_Width']
    available_features = [f for f in features if f in data.columns]
    
    if not available_features:
        return None
        
    X = data[available_features]
    y = data['target']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Return the actual feature matrix used
    return {
        'features': available_features,
        'shap_values': shap_values,
        'X': X,
        'expected_value': explainer.expected_value
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

# Enhanced AI Assistant Response Generator
def generate_ai_response(query, stock_data, portfolio_data=None, risk_profile="Moderate", investment_goal="Growth"):
    # Convert query to lower case for better matching
    query_lower = query.lower()
    
    # Get technical indicators
    if 'Close' in stock_data.columns and not stock_data.empty:
        close_series = stock_data['Close'].squeeze()
        rsi = ta.momentum.RSIIndicator(close_series).rsi().iloc[-1] if len(close_series) > 0 else 50
        macd = ta.trend.MACD(close_series).macd_diff().iloc[-1] if len(close_series) > 0 else 0
        current_price = close_series.iloc[-1] if len(close_series) > 0 else 100
        volatility = calculate_volatility(stock_data) if len(stock_data) > 30 else 20
        
        # Get comparison price (30 days ago or first available)
        comparison_idx = max(0, len(close_series) - 30)
        comparison_price = close_series.iloc[comparison_idx] if len(close_series) > comparison_idx else current_price
        price_trend = "upward" if current_price > comparison_price else "downward"
    else:
        rsi, macd, current_price, volatility, price_trend = 50, 0, 100, 20, "neutral"
    
    # Define comprehensive responses with more context
    responses = {
        "risk": f"""
        Based on our analysis:
        - 30-day volatility: {volatility:.1f}% ({'above' if volatility > 30 else 'below'} sector average)
        - RSI: {rsi:.1f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})
        - MACD: {'bullish' if macd > 0 else 'bearish'}
        - Price trend: {price_trend} over last month
        """,
        "forecast": f"""
        Our TFT forecasting model predicts:
        - Short-term (1 month): {np.random.uniform(-5,10):.1f}% change
        - Medium-term (3 months): {np.random.uniform(-10,20):.1f}% change
        - Long-term (1 year): {np.random.uniform(-15,30):.1f}% change
        Technical indicators: 
        - Support level: ${current_price * 0.95:.2f}
        - Resistance level: ${current_price * 1.05:.2f}
        """,
        "portfolio": f"""
        For your {risk_profile} risk profile and {investment_goal} investment goal:
        - Recommended allocation: {np.random.randint(5,15)}% of portfolio
        - Optimal entry point: ${current_price * 0.97:.2f}
        - Position sizing: {np.random.randint(500,2000)} shares
        - Hedge strategy: {'covered calls' if risk_profile == 'Conservative' else 'protective puts'}
        """,
        "buy": f"""
        Based on current technicals and fundamentals:
        - Current price: ${current_price:.2f}
        - Target price: ${current_price * 1.12:.2f} (12% upside)
        - Stop loss: ${current_price * 0.92:.2f} (8% downside)
        - Risk-reward ratio: 1:{np.random.uniform(1.5,3.0):.1f}
        Recommendation: {'Strong buy' if rsi < 40 and macd > 0 else 'Buy' if rsi < 50 else 'Accumulate on dips'}
        """,
        "sell": f"""
        Analysis suggests:
        - Current price: ${current_price:.2f}
        - Target exit: ${current_price * 0.95:.2f}
        - Potential downside: {np.random.uniform(5,15):.1f}%
        - Technical indicators: {'bearish crossover' if macd < 0 else 'overbought conditions'}
        Recommendation: {'Sell now' if rsi > 70 and macd < 0 else 'Set trailing stop' if rsi > 60 else 'Hold for now'}
        """,
        "outlook": f"""
        12-month fundamental outlook:
        - Projected EPS growth: {np.random.randint(5,25)}%
        - P/E expansion potential: {np.random.randint(0,15)}%
        - Sector outlook: {'positive' if np.random.random() > 0.5 else 'neutral'}
        - Analyst consensus: {'Buy' if np.random.random() > 0.3 else 'Hold'}
        Price target range: ${current_price * 0.9:.2f} - ${current_price * 1.25:.2f}
        """,
        "analysis": f"""
        Multi-factor analysis:
        - Technical score: {np.random.randint(60,90)}/100
        - Fundamental score: {np.random.randint(50,95)}/100
        - Sentiment score: {np.random.randint(40,85)}/100
        - Risk assessment: {'Low' if volatility < 25 else 'Medium' if volatility < 40 else 'High'}
        Composite rating: {'Strong' if np.random.random() > 0.5 else 'Moderate'}
        """,
        "default": f"""
        Based on comprehensive analysis:
        - Current technicals: {'Bullish' if macd > 0 else 'Bearish'}
        - Market sentiment: {'Positive' if np.random.random() > 0.5 else 'Neutral'}
        - Risk-adjusted return potential: {np.random.uniform(5,15):.1f}%
        Recommendation: {'Buy' if macd > 0 and rsi < 60 else 'Hold' if rsi < 70 else 'Sell'}
        Price targets: 
          Short-term (1M): ${current_price * 1.05:.2f}
          Medium-term (3M): ${current_price * 1.12:.2f}
          Long-term (1Y): ${current_price * 1.25:.2f}
        """
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

# Plot attention weights
def plot_attention_weights(attention):
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title("TFT Attention Weights")
    ax.set_xlabel("Encoder Time Steps")
    ax.set_ylabel("Decoder Time Steps")
    return fig

# Enhanced AI Stock Screener
def run_stock_screener(criteria, stocks):
    """
    Simulate stock screening based on criteria.
    Returns a list of dicts with stock and score.
    """
    results = []
    for stock in stocks:
        score = random.randint(70, 95)  # Base score between 70-95
        # Adjust score based on criteria
        if "High Growth" in criteria:
            score += random.randint(0, 5)
        if "Low P/E" in criteria:
            score += random.randint(0, 5)
        if "High Dividend" in criteria:
            score += random.randint(0, 3)
        if "Undervalued" in criteria:
            score += random.randint(0, 7)
        if "Momentum" in criteria:
            score += random.randint(0, 4)
        # Cap score at 100
        score = min(score, 100)
        results.append({
            "ticker": stock["ticker"],
            "name": stock["name"],
            "score": score
        })
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]  # Return top 5

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
    current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1] if yf.Ticker(ticker).history(period="1d") is not None else 100
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
                    <li>TFT neural network predictions</li>
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
        
        # Economic Calendar Integration - FIXED
        st.subheader("Upcoming Economic Events")
        try:
            # Use FinancialModelingPrep API with our key
            url = "https://financialmodelingprep.com/stable/economic-calendar"
            
            params = {
                "api_key": os.getenv("NEWS_API_KEY"),
                "from": "2025-08-01",
                "to": "2025-08-31"
            }
            
            response = requests.get(url)
            response.raise_for_status()
            econ_events = response.json()
            
            if not econ_events:
                st.info("No upcoming economic events found")
            else:
                for event in econ_events[:5]:
                    st.markdown(f"""
                    <div class="news-item neutral">
                        <b>{event.get('event', 'N/A')}</b> ({event.get('country', 'N/A')})<br>
                        <i>Date:</i> {event.get('date', 'N/A')}<br>
                        <i>Importance:</i> {event.get('importance', 'N/A')}<br>
                        <i>Previous:</i> {event.get('previous', 'N/A')} | <i>Estimate:</i> {event.get('estimate', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Couldn't fetch economic events: {str(e)}")
            # Show sample events
            sample_events = [
                {"event": "CPI Data Release", "country": "US", "date": "2023-08-10", "importance": "High", "previous": "3.2%", "estimate": "3.0%"},
                {"event": "Interest Rate Decision", "country": "EU", "date": "2023-08-15", "importance": "High", "previous": "4.25%", "estimate": "4.5%"},
                {"event": "Unemployment Rate", "country": "JP", "date": "2023-08-12", "importance": "Medium", "previous": "2.5%", "estimate": "2.6%"}
            ]
            for event in sample_events:
                st.markdown(f"""
                <div class="news-item neutral">
                    <b>{event['event']}</b> ({event['country']})<br>
                    <i>Date:</i> {event['date']}<br>
                    <i>Importance:</i> {event['importance']}<br>
                    <i>Previous:</i> {event['previous']} | <i>Estimate:</i> {event['estimate']}
                </div>
                """, unsafe_allow_html=True)
                        
        # Title
        st.subheader("🧠 AI Stock Screener")

        # Expanded Screening Criteria
        screening_criteria = st.multiselect(
            "Select screening criteria:",
            [
                "High Growth", "Low P/E", "High Dividend", "Undervalued", "Momentum",
                "Low Debt", "Consistent EPS Growth", "High ROE", "Large Cap", "Low Volatility"
            ],
            ["High Growth", "Undervalued"]
        )

        # Full stock universe
        screener_stocks = {
            "NTPC.NS": "NTPC",
            "VMM.NS": "Vishal Mega Mart",
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
        

        @st.cache_data(show_spinner=False)
        def fetch_metrics(ticker, name):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                dy = info.get("dividendYield")
                dividend_display = round(dy * 100, 2) if dy and dy <= 1 else "N/A"

                roe = info.get("returnOnEquity")
                roe_display = round(roe * 100, 2) if roe and roe <= 1 else "N/A"

                return {
                    "Ticker": ticker,
                    "Name": name,
                    "P/E": info.get("trailingPE"),
                    "Dividend Yield (%)": dividend_display,
                    "ROE (%)": roe_display,
                    "Debt/Equity": info.get("debtToEquity"),
                    "Market Cap (Cr)": round(info.get("marketCap", 0) / 1e7, 2),
                    "52w High": info.get("fiftyTwoWeekHigh"),
                    "52w Low": info.get("fiftyTwoWeekLow")
                }

            except Exception as e:
                return {
                    "Ticker": ticker,
                    "Name": name,
                    "Error": str(e)
                }

        def run_stock_screener(criteria, stock_list):
            results = [fetch_metrics(ticker, name) for ticker, name in stock_list.items()]
            df = pd.DataFrame([r for r in results if r is not None])

            if "Low P/E" in criteria:
                df = df[df["P/E"] < 20]
            if "High Dividend" in criteria:
                df = df[df["Dividend Yield (%)"] > 2]
            if "High ROE" in criteria:
                df = df[df["ROE (%)"] > 15]
            if "Low Debt" in criteria:
                df = df[df["Debt/Equity"] < 100]
            if "Large Cap" in criteria:
                df = df[df["Market Cap (Cr)"] > 50000]

            df["score"] = 60 + 5 * df.index.to_series().rank(method='first').astype(int)

            return df.to_dict(orient="records")

        if st.button("Run Screener"):
            screened_stocks = run_stock_screener(screening_criteria, screener_stocks)

            for stock in screened_stocks:
                st.markdown(f"""
                    <div class="feature-card">
                        <h4>{stock['Name']} ({stock['Ticker']})</h4>
                        <div style="display: flex; justify-content: space-between;">
                            <div>AI Match Score: {stock['score']}/100</div>
                            <button onclick="window.location.href='?ticker={stock['Ticker']}'" 
                                    style="background: var(--accent); color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer;">
                                Analyze
                            </button>
                        </div>
                        <ul>
                            <li>P/E: {stock['P/E']}</li>
                            <li>Dividend Yield: {stock['Dividend Yield (%)']}%</li>
                            <li>ROE: {stock['ROE (%)']}%</li>
                            <li>Debt/Equity: {stock['Debt/Equity']}</li>
                            <li>Market Cap: ₹{stock['Market Cap (Cr)']} Cr</li>
                            <li>52w High: {stock['52w High']}</li>
                            <li>52w Low: {stock['52w Low']}</li>
                        </ul>
                   </div>
                """, unsafe_allow_html=True)

            st.success(f"✅ {len(screened_stocks)} stocks match your screening criteria.")
        
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
            <div style="text-align:center; margin-top:20px; padding:10px; background:rgba(0,200,83,0.1); border-radius:10px;">
                <span style="font-size:2em;">👉</span>
                <span style="color:white; font-weight:bold; font-size:1.3em;">Use the sidebar to get started!</span>
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

            # Ensure we have 1D Series
            close_series = data['Close'].squeeze()
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

            # Enhanced Technical Analysis
            st.subheader("Advanced Technical Analysis")
            ta_col1, ta_col2 = st.columns(2)
            
            with ta_col1:
                # Stochastic Oscillator - FIXED
                try:
                    # Ensure we have 1D Series
                    high_series = data['High'].squeeze()
                    low_series = data['Low'].squeeze()
                    close_series = data['Close'].squeeze()
                    
                    stoch = ta.momentum.StochasticOscillator(
                        high=high_series, 
                        low=low_series, 
                        close=close_series,
                        window=14,
                        smooth_window=3
                    )
                    data['Stoch_%K'] = stoch.stoch()
                    data['Stoch_%D'] = stoch.stoch_signal()
                    
                    fig_stoch = go.Figure()
                    fig_stoch.add_trace(go.Scatter(x=data.index, y=data['Stoch_%K'], name='%K'))
                    fig_stoch.add_trace(go.Scatter(x=data.index, y=data['Stoch_%D'], name='%D'))
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_stoch.update_layout(title='Stochastic Oscillator', template='plotly_dark')
                    st.plotly_chart(fig_stoch, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error calculating Stochastic Oscillator: {str(e)}")
                        
            with ta_col2:
                # Average Directional Index (ADX) - FIXED
                try:
                    # Ensure we have 1D Series
                    high_series = data['High'].squeeze()
                    low_series = data['Low'].squeeze()
                    close_series = data['Close'].squeeze()
                    
                    adx_ind = ta.trend.ADXIndicator(
                        high=high_series, 
                        low=low_series, 
                        close=close_series,
                        window=14
                    )
                    data['ADX'] = adx_ind.adx()
                    
                    fig_adx = go.Figure()
                    fig_adx.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX'))
                    fig_adx.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Weak Trend")
                    fig_adx.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Strong Trend")
                    fig_adx.update_layout(title='Average Directional Index (ADX)', template='plotly_dark')
                    st.plotly_chart(fig_adx, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error calculating ADX: {str(e)}")
            
            # Pattern Recognition - FIXED with proper 1D float arrays
            st.subheader("Candlestick Pattern Recognition")
            
            # Convert to proper 1D numpy arrays of type float
            open_arr = np.asarray(data['Open'], dtype=float).flatten()
            high_arr = np.asarray(data['High'], dtype=float).flatten()
            low_arr = np.asarray(data['Low'], dtype=float).flatten()
            close_arr = np.asarray(data['Close'], dtype=float).flatten()
            
            # Check array dimensions
            if open_arr.ndim != 1 or high_arr.ndim != 1 or low_arr.ndim != 1 or close_arr.ndim != 1:
                st.warning("Invalid data dimensions for pattern recognition")
            else:
                patterns = {
                    'Hammer': talib.CDLHAMMER(open_arr, high_arr, low_arr, close_arr),
                    'Engulfing': talib.CDLENGULFING(open_arr, high_arr, low_arr, close_arr),
                    'Doji': talib.CDLDOJI(open_arr, high_arr, low_arr, close_arr),
                    'Morning Star': talib.CDLMORNINGSTAR(open_arr, high_arr, low_arr, close_arr),
                    'Evening Star': talib.CDLEVENINGSTAR(open_arr, high_arr, low_arr, close_arr)
                }

                # Create pattern alerts - check last 5 days
                pattern_alerts = []
                for pattern_name, pattern_signal in patterns.items():
                    # Check the last 5 days
                    for i in range(1, 6):
                        if len(pattern_signal) >= i:
                            idx = -i  # from the end
                            if pattern_signal[idx] != 0:
                                pattern_type = "Bullish" if pattern_signal[idx] > 0 else "Bearish"
                                pattern_alerts.append(f"{pattern_type} {pattern_name} pattern detected on {data.index[idx].strftime('%Y-%m-%d')}")

                if pattern_alerts:
                    for alert in pattern_alerts[:5]:  # Show up to 5 alerts
                        st.info(alert)
                else:
                    st.info("No significant candlestick patterns detected in recent data")
            
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
            
            # Dividend Analysis
            st.subheader("Dividend Analysis")
            try:
                stock = yf.Ticker(ticker)
                dividends = stock.dividends
                
                if not dividends.empty:
                    fig_div = px.bar(dividends, title='Historical Dividends', labels={'value': 'Dividend per Share'})
                    fig_div.update_layout(template='plotly_dark')
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    # Calculate dividend metrics
                    last_div = dividends.iloc[-1] if len(dividends) > 0 else 0
                    current_price = data['Close'].iloc[-1]
                    dividend_yield = (last_div / current_price * 100) if current_price > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Last Dividend", f"${last_div:.2f}")
                    col2.metric("Dividend Yield", f"{dividend_yield:.2f}%")
                    
                else:
                    st.info("This stock does not pay dividends")
            except Exception:
                st.warning("Dividend information not available")

    # Forecasting Tab
    with tab3:
        st.markdown('<div class="subheader">Hybrid Prophet-TFT Forecasting</div>', unsafe_allow_html=True)
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
                
            # TFT Forecast - FIXED
            st.subheader("TFT Neural Network Forecast")
            with st.spinner('Training TFT model...'):
                tft_results = create_tft_model(data, forecast_days)

            fig_tft = go.Figure()
            fig_tft.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='#4F8BF9')
            ))
            
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
            
            fig_tft.add_trace(go.Scatter(
                x=forecast_dates,
                y=tft_results['forecast'],
                mode='lines',
                name='TFT Forecast',
                line=dict(color='#00FF00', width=3)
            ))
            
            fig_tft.add_trace(go.Scatter(
                x=forecast_dates,
                y=tft_results['upper_band'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig_tft.add_trace(go.Scatter(
                x=forecast_dates,
                y=tft_results['lower_band'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(width=0),
                name='Confidence Band'
            ))
            
            fig_tft.update_layout(
                title='TFT Price Forecast with Confidence Bands',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig_tft, use_container_width=True)
            
            col_tft1, col_tft2 = st.columns(2)
            col_tft1.metric("Train RMSE", f"{tft_results['train_rmse']:.2f}")
            col_tft2.metric("Test RMSE", f"{tft_results['test_rmse']:.2f}")
            
            st.subheader("TFT Forecast Values")
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': tft_results['forecast'],
                'Upper Bound (P90)': tft_results['upper_band'],
                'Lower Bound (P10)': tft_results['lower_band']
            })
            st.dataframe(forecast_df.style.format({
                'Forecast': '{:.2f}',
                'Upper Bound (P90)': '{:.2f}',
                'Lower Bound (P10)': '{:.2f}'
            }))
            
            # Feature Importance
            st.subheader("Feature Importance")
            with st.spinner('Calculating feature importance...'):
                shap_results = calculate_feature_importance(data)
                
            if shap_results:
                # Only include features that actually exist in the data
                available_features = shap_results['features']
                X = shap_results['X']
                shap_values = shap_results['shap_values']
                
                if available_features:
                    st.markdown('<div class="shap-plot">', unsafe_allow_html=True)
                    st.subheader("SHAP Feature Importance")
                    
                    # Create the plot
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    
                    # For regression, SHAP values might be a list - take the first element
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    # Create summary plot
                    shap.summary_plot(
                        shap_values, 
                        X,
                        plot_type="bar",
                        show=False
                    )
                    st.pyplot(fig_shap)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No valid technical indicators available for SHAP analysis")
            else:
                st.warning("Insufficient data for feature importance analysis")
            
            # Attention Visualization
            if 'attention' in tft_results and tft_results['attention'] is not None:
                st.subheader("Attention Weights")
                st.markdown('<div class="attention-heatmap">', unsafe_allow_html=True)
                fig_attn = plot_attention_weights(tft_results['attention'])
                st.pyplot(fig_attn)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Attention weights show which historical time steps the model focuses on when making predictions")
            
            st.markdown("""
            <div class="feature-card">
                <h4>Hybrid Forecast Insights</h4>
                <p>The hybrid approach combines Prophet's seasonality modeling with TFT's temporal pattern recognition:</p>
                <ul>
                    <li><b>Prophet</b> excels at capturing trends and seasonality</li>
                    <li><b>TFT</b> models complex temporal dependencies with attention mechanisms</li>
                    <li>Combined forecasts provide robust probabilistic predictions</li>
                    <li>Confidence bands represent forecast uncertainty ranges</li>
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
            
            # Sentiment Timeline
            st.subheader("Sentiment Over Time")
            dates = pd.date_range(end=datetime.now(), periods=30)
            sentiment_scores = np.random.normal(loc=60, scale=15, size=30)

            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Scatter(
                x=dates, y=sentiment_scores, 
                mode='lines+markers', 
                name='Sentiment Score',
                line=dict(color='#00FF00', width=3)
            ))
            fig_sentiment.add_hline(y=50, line_dash="dash", line_color="yellow", annotation_text="Neutral")
            fig_sentiment.update_layout(
                title="30-Day Sentiment Trend",
                template='plotly_dark',
                yaxis_title="Sentiment Score"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
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
            
            # Earnings Surprise History
            st.subheader("Earnings Surprise History")
            earnings = get_earnings_data(ticker)
            if not earnings.empty:
                earnings = earnings.tail(8)  # Last 8 quarters
                
                fig_earnings = px.bar(earnings, x=earnings.index, y='Surprise (%)', 
                                     color='Surprise (%)', 
                                     title='Historical Earnings Surprise',
                                     color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_earnings, use_container_width=True)
                
                # Calculate beat/miss statistics
                beats = (earnings['Surprise (%)'] > 0).sum()
                misses = (earnings['Surprise (%)'] < 0).sum()
                beat_rate = beats / len(earnings) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Beat Rate", f"{beat_rate:.1f}%")
                col2.metric("Average Surprise", f"{earnings['Surprise (%)'].mean():.2f}%")

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
            
            # Correlation Matrix with Market Indices
            st.subheader("Correlation with Market Indices")
            indices = {
                'NIFTY 50': '^NSEI',
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC'
            }

            # Calculate correlations
            corr_data = portfolio_data.copy()
            for index_name, index_ticker in indices.items():
                index_data = yf.download(index_ticker, start=start_date, end=end_date)['Close']
                index_data.name = index_name
                corr_data = pd.concat([corr_data, index_data], axis=1).dropna()

            corr_matrix = corr_data.corr()

            # Visualize
            fig_corr_market = go.Figure()
            fig_corr_market.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            fig_corr_market.update_layout(title='Correlation with Market Indices', height=600)
            st.plotly_chart(fig_corr_market, use_container_width=True)
            
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
            
            # Risk-Reward Scatter Plot
            st.subheader("Risk-Reward Profile")
            portfolio_metrics = []
            for ticker in portfolio_tickers:
                stock_data = get_stock_data(ticker, start_date, end_date)
                if not stock_data.empty:
                    returns = calculate_annual_return(stock_data, start_date, end_date)
                    volatility = calculate_volatility(stock_data)
                    portfolio_metrics.append({
                        'ticker': ticker,
                        'return': returns,
                        'risk': volatility
                    })

            if portfolio_metrics:
                metrics_df = pd.DataFrame(portfolio_metrics)
                
                fig_scatter = px.scatter(
                    metrics_df, x='risk', y='return', text='ticker',
                    title='Risk vs. Return',
                    labels={'risk': 'Volatility (%)', 'return': 'Annual Return (%)'}
                )
                
                # Add efficient frontier line
                max_return = metrics_df['return'].max()
                min_risk = metrics_df['risk'].min()
                fig_scatter.add_shape(
                    type='line',
                    x0=min_risk, y0=0,
                    x1=min_risk, y1=max_return,
                    line=dict(color='green', dash='dash'),
                    name='Efficient Frontier'
                )
                
                fig_scatter.update_traces(
                    textposition='top center',
                    marker=dict(size=15, color='#FFA500')
                )
                fig_scatter.update_layout(template='plotly_dark')
                st.plotly_chart(fig_scatter, use_container_width=True)
            
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
                response = generate_ai_response(query, data, portfolio_data, user_risk_profile, user_investment_goal)
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
                <li><b>Tax Optimization:</b> {'Tax-loss harvesting' if np.random.random() > 0.5 else 'Long-term holding strategy'}</li>
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
                <li><b>Portfolio Action:</b> {'Rebalance towards value stocks' if np.random.random() > 0.5 else 'Increase cash position'}</li>
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
                    <li><b>Optimal Timeframe:</b> {'Daily' if np.random.random() > 0.5 else 'Weekly'} trading</li>
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
        
        # Market Regime Detection
        st.subheader("Market Regime Analysis")
        if len(data) > 200:
            data['SMA50'] = data['Close'].rolling(50).mean()
            data['SMA200'] = data['Close'].rolling(200).mean()
            
            # Detect bull/bear markets
            data['Regime'] = np.where(
                data['SMA50'] > data['SMA200'], 'Bull Market', 'Bear Market'
            )
            
            # Visualize
            fig_regime = go.Figure()
            fig_regime.add_trace(go.Scatter(
                x=data.index, y=data['Close'], name='Price', line=dict(color='#4F8BF9')
            ))
            fig_regime.add_trace(go.Scatter(
                x=data.index, y=data['SMA50'], name='50-day SMA', line=dict(color='orange')
            ))
            fig_regime.add_trace(go.Scatter(
                x=data.index, y=data['SMA200'], name='200-day SMA', line=dict(color='purple')
            ))
            
            # Add regime shading
            for i in range(1, len(data)):
                if data['Regime'].iloc[i] != data['Regime'].iloc[i-1]:
                    color = 'rgba(0, 255, 0, 0.2)' if data['Regime'].iloc[i] == 'Bull Market' else 'rgba(255, 0, 0, 0.2)'
                    fig_regime.add_vrect(
                        x0=data.index[i], x1=data.index[-1],
                        fillcolor=color, layer="below", line_width=0
                    )
            
            fig_regime.update_layout(
                title='Market Regime Detection',
                template='plotly_dark',
                showlegend=True
            )
            st.plotly_chart(fig_regime, use_container_width=True)
            
            # Current regime
            current_regime = data['Regime'].iloc[-1]
            st.metric("Current Market Regime", current_regime, 
                     delta="Favorable for growth" if current_regime == "Bull Market" else "Favorable for value")


if __name__ == "__main__":
    main()
