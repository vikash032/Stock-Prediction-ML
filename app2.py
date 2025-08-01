# Critical fix for NameError
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("NEWS_API_KEY")

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
import re
import ta  # Technical analysis library
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 2 minutes
st_autorefresh(interval=120000, key="data_refresh")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .header { font-size: 2.5em; color: #4F8BF9; text-align: center; margin-bottom: 30px; }
    .subheader { font-size: 1.5em; color: #4F8BF9; border-bottom: 2px solid #4F8BF9; padding-bottom: 10px; margin-top: 20px; }
    .metric-card { background-color: #1e1e2f; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .stButton>button { background-color: #4F8BF9 !important; color: white !important; border-radius: 4px !important; }
    .news-item {
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 6px;
        font-size: 0.95rem;
        font-weight: 500;
        background-color: #1c1c2e;
    }
    .positive {
        border-left: 6px solid #4CAF50;
        background-color: #2e7d32 !important;
        color: #e8f5e9 !important;
    }
    .negative {
        border-left: 6px solid #F44336;
        background-color: #b71c1c !important;
        color: #ffebee !important;
    }
    .neutral {
        border-left: 6px solid #2196F3;
        background-color: #283593 !important;
        color: #e3f2fd !important;
    }
    .news-item a {
        color: #FFD700;
        font-weight: bold;
        text_decoration: underline;
    }
    .improvement-card {
        background-color: #1c1c2e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4F8BF9;
    }
    .gauge {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, #e74c3c 0%, #f1c40f 50%, #2ecc71 100%);
    }
    .gauge-value {
        font-size: 1.5rem;
        font-weight: bold;
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
        "ZOMATO.NS": "Zomato",
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

# ------------------ MAIN APP START ------------------
def main():
    st.markdown('<h1 class="header">📊 Advanced Stock Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.sidebar.header("Configuration")
    default_tickers = [
        "NTPC.NS", "VMM.NS", "ZOMATO.NS", "SAGILITY.NS", "TATAMOTORS.NS",
        "TCS.NS", "SBIN.NS", "KALYANKJIL.NS", "SWANENERGY.NS", "PRAJIND.NS",
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
        "BAJFINANCE.NS", "LT.NS", "AXISBANK.NS", "ADANIENT.NS", "BHARTIARTL.NS",
        "HCLTECH.NS", "KOTAKBANK.NS", "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS"
    ]

    ticker = st.sidebar.selectbox("Select Stock", default_tickers, index=0)
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)
    risk_tolerance = st.sidebar.slider("Risk Tolerance (1=Low, 10=High)", 1, 10, 5)
    portfolio_size = st.sidebar.number_input("Portfolio Size ($)", 10000, 1000000, 50000)
    portfolio_tickers = st.sidebar.multiselect("Select Portfolio Stocks", default_tickers, default=default_tickers)
    
    # Add market sentiment gauge
    st.sidebar.markdown("### Market Sentiment")
    sentiment_value = st.sidebar.slider("Bull/Bear Indicator", 0, 100, 65)
    st.sidebar.markdown(f"""
        <div class="gauge">
            <div class="gauge-value">{sentiment_value}/100</div>
            <small>{'Bullish' if sentiment_value > 60 else 'Bearish' if sentiment_value < 40 else 'Neutral'} Market</small>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner('Fetching market data...'):
        data = get_stock_data(ticker, start_date, end_date)

    if data.empty:
        st.error(f"No data available for {ticker}. Please try a different ticker.")
        return

    # Create tabs with Home as the first tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Market Data", "Forecasting", "Sentiment Analysis", "Portfolio Optimization"])

    # Home Tab Content
    with tab1:
            st.markdown('<div class="subheader" style="color:#FFD700; font-size:2em;">🚀 Welcome to Advanced Stock Analytics</div>', unsafe_allow_html=True)

            # Project Introduction
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #1a2a6c, #003366); border-left: 6px solid #FFD700;">
                <h3 style="color:#FFD700;">📊 Project Overview</h3>
                <p style="color:#FFFFFF; font-size:1.1em;">Advanced Stock Analytics is a comprehensive financial analysis platform that combines real-time market data, 
                predictive forecasting, sentiment analysis, and portfolio optimization to empower investors with actionable insights.</p>
            </div>
            """, unsafe_allow_html=True)

            # Key Features Section
            st.markdown('<div class="subheader" style="color:#00FF7F; border-bottom: 2px solid #00FF7F;">✨ Key Features</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="improvement-card" style="background: linear-gradient(135deg, #0f2027, #203a43); border-left: 4px solid #FF8C00;">
                    <h4 style="color:#FF8C00;">📈 Real-Time Market Intelligence</h4>
                    <ul style="color:#E0FFFF;">
                        <li>Live price tracking with candlestick charts</li>
                        <li>Technical indicators (RSI, MACD, Moving Averages)</li>
                        <li>Volatility and return metrics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="improvement-card" style="background: linear-gradient(135deg, #23074d, #cc5333); border-left: 4px solid #BA55D3;">
                    <h4 style="color:#BA55D3;">🔮 Hybrid Forecasting</h4>
                    <ul style="color:#E0FFFF;">
                        <li>Prophet time-series forecasting</li>
                        <li>Confidence interval projections</li>
                        <li>Risk assessment metrics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="improvement-card" style="background: linear-gradient(135deg, #00416A, #799F0C); border-left: 4px solid #7CFC00;">
                    <h4 style="color:#7CFC00;">💹 Portfolio Optimization</h4>
                    <ul style="color:#E0FFFF;">
                        <li>Modern Portfolio Theory (MPT) implementation</li>
                        <li>Risk-adjusted allocation strategies</li>
                        <li>Monte Carlo simulations</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Unique Features Section
            st.markdown('<div class="subheader" style="color:#FF6347; border-bottom: 2px solid #FF6347;">💎 Unique Features</div>', unsafe_allow_html=True)

            unique_col1, unique_col2 = st.columns([2, 1])

            with unique_col1:
                st.markdown("""
                <div class="improvement-card" style="background: linear-gradient(135deg, #4b1248, #f0c27b); border-left: 4px solid #FF1493;">
                    <h4 style="color:#FF1493;">🧠 Sentiment-Driven Analysis</h4>
                    <p style="color:#FFFAF0;">Our proprietary sentiment engine combines:</p>
                    <ul style="color:#FFFAF0;">
                        <li>FinBERT financial sentiment analysis model</li>
                        <li>Real-time news aggregation from global sources</li>
                        <li>Bull/Bear market sentiment gauge</li>
                        <li>Sentiment-weighted risk assessment</li>
                    </ul>
                </div>

                <div class="improvement-card" style="background: linear-gradient(135deg, #141E30, #243B55); border-left: 4px solid #00FFFF;">
                    <h4 style="color:#00FFFF;">⚡ Adaptive Portfolio Engine</h4>
                    <ul style="color:#E0FFFF;">
                        <li>Dynamic risk tolerance scaling (1-10)</li>
                        <li>Correlation heatmap visualization</li>
                        <li>Return comparison metrics</li>
                        <li>Monte Carlo simulation for risk profiling</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with unique_col2:
                st.markdown("""
                <div class="metric-card" style="text-align:center; background:linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c); border: 2px solid #FFD700; padding:20px;">
                    <h3 style="color:#FFD700; font-size:1.5em;">Tech Stack</h3>
                    <div style="font-size:3rem;">🤖</div>
                    <p style="color:#FFFFFF; font-weight:bold;">AI-Powered Analytics</p>
                    <ul style="text-align:left; color:#FFD700;">
                        <li>Prophet Forecasting</li>
                        <li>FinBERT NLP</li>
                        <li>CVXPY Optimization</li>
                    </ul>
                    <p style="color:#FFFFFF; font-weight:bold;">Real-Time Data</p>
                    <ul style="text-align:left; color:#FFD700;">
                        <li>Yahoo Finance API</li>
                        <li>NewsAPI Integration</li>
                        <li>Streamlit Live Updates</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Usage Instructions
            st.markdown('<div class="subheader" style="color:#00BFFF; border-bottom: 2px solid #00BFFF;">🚦 Getting Started</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="improvement-card" style="background: linear-gradient(135deg, #000428, #004e92); border-left: 4px solid #00FF00;">
                <ol style="color:#FFFFFF; font-size:1.1em;">
                    <li><b style="color:#00FF00;">Select a stock</b> from the sidebar dropdown</li>
                    <li><b style="color:#00FF00;">Adjust date ranges</b> and forecast periods</li>
                    <li><b style="color:#00FF00;">Explore different tabs</b> for various analyses</li>
                    <li><b style="color:#00FF00;">Build portfolios</b> with multiple stocks</li>
                    <li><b style="color:#00FF00;">Adjust risk tolerance</b> for personalized optimization</li>
                </ol>
                <div style="text-align:center; margin-top:20px;">
                    <span style="font-size:2em;">👉</span>
                    <span style="color:#FFD700; font-weight:bold; font-size:1.2em;">Use the sidebar to get started!</span>
                    <span style="font-size:2em;">👈</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


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
                <div class="metric-card" style="background-color: #0f2d21; border-left: 6px solid #2ecc71; color: #ffffff;">
                    <b>Current Price</b><br>${current_price:.2f}
                </div>''', unsafe_allow_html=True)
            col2.markdown(f'''
                <div class="metric-card" style="background-color: #322406; border-left: 6px solid #e67e22; color: #ffffff;">
                    <b>Daily Change</b><br>{daily_change:.2f}%
                </div>''', unsafe_allow_html=True)
            col3.markdown(f'''
                <div class="metric-card" style="background-color: #271535; border-left: 6px solid #9b59b6; color: #ffffff;">
                    <b>Annual Volatility</b><br>{volatility:.2f}%
                </div>''', unsafe_allow_html=True)
            col4.markdown(f'''
                <div class="metric-card" style="background-color: #0b2940; border-left: 6px solid #3498db; color: #ffffff;">
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

            # Calculate RSI
            close_series = data['Close'].squeeze()
            data['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()

            # Calculate MACD
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
            
            # Recent data table
            st.subheader("Recent Price Data")
            st.dataframe(data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].style.format({
                'Open': '{:.2f}', 'High': '{:.2f}',
                'Low': '{:.2f}', 'Close': '{:.2f}',
                'Volume': '{:,}', 'RSI': '{:.2f}',
                'MACD': '{:.4f}'
            }), height=400)
    
    with tab3:
        st.markdown('<div class="subheader">Hybrid Prophet-LSTM Forecasting</div>', unsafe_allow_html=True)
        if len(data) < 60:
            st.warning("Need at least 60 days of data for forecasting")
            st.stop()

        with st.spinner('Training forecasting models...'):
            # Prophet Forecast
            prophet_df = data[['Close']].reset_index()
            prophet_df.columns = ['ds', 'y']
            model = Prophet(
                daily_seasonality=False, 
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10
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
                
                # Show forecast summary
                st.subheader("Forecast Summary")
                forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                st.dataframe(forecast[forecast_cols].tail(10).rename(columns={
                    'ds': 'Date', 'yhat': 'Forecast',
                    'yhat_lower': 'Low', 'yhat_upper': 'High'
                }).style.format({
                    'Forecast': '{:.2f}', 'Low': '{:.2f}', 'High': '{:.2f}'
                }))
                
                # Risk assessment
                last_forecast = forecast.iloc[-1]
                confidence_interval = last_forecast['yhat_upper'] - last_forecast['yhat_lower']
                confidence_percent = min(100, max(0, 100 - (confidence_interval / last_forecast['yhat'] * 100)))
                
                st.metric("Forecast Confidence", f"{confidence_percent:.1f}%")
                st.progress(int(confidence_percent))
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")

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
                'Allocation ($)': [w * portfolio_size for w in weights]
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

if __name__ == "__main__":
    main()
