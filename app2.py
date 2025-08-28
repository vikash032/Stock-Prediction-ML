import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ta
import warnings
from datetime import datetime, timedelta
import logging
import requests
from groq import Groq
import os
from dotenv import load_dotenv
import re

# ------------------ CONFIGURATION ------------------
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Quantum Stock Prediction - India",
    page_icon="📈",
    layout="wide"
)

# ------------------ DATA MODULE ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance with enhanced error handling"""
    try:
        logger.info(f"Fetching data for {ticker} from {start} to {end}")
        
        # Validate date range
        if start >= end:
            st.error("Start date must be before end date.")
            return pd.DataFrame()
            
        # Extend the start date to ensure we have enough data for calculations
        extended_start = start - timedelta(days=60)
        
        # Try multiple times with different parameters if needed
        data = yf.download(ticker, start=extended_start, end=end, progress=False, auto_adjust=True)
        
        if data.empty:
            logger.warning(f"No data found for {ticker} with auto_adjust=True, trying without auto_adjust")
            # Try without auto_adjust
            data = yf.download(ticker, start=extended_start, end=end, progress=False, auto_adjust=False)
            
        if data.empty:
            # Try with a shorter period to validate the ticker
            test_data = yf.download(ticker, period="1mo", progress=False)
            if test_data.empty:
                logger.error(f"Invalid ticker symbol: {ticker}")
                st.error(f"Invalid ticker symbol: {ticker}. Please check the symbol and try again.")
            else:
                logger.error(f"No data available for {ticker} in the selected date range")
                st.error(f"No data available for {ticker} in the selected date range. Try a different date range.")
            return pd.DataFrame()
            
        return data
    except Exception as e:
        logger.error(f"Data fetch error: {str(e)}")
        st.error(f"Data fetch error: {str(e)}. Please try a different ticker or date range.")
        return pd.DataFrame()

# ------------------ NEWS SENTIMENT MODULE ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_news(ticker, num_articles=5):
    """Fetch news articles related to a stock"""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        st.warning("News API key not found. Using sample news data.")
        return get_sample_news(ticker, num_articles)
    
    # Map tickers to company names for better search
    company_map = {
        "RELIANCE.NS": "Reliance Industries", 
        "TATAMOTORS.NS": "Tata Motors",
        "TCS.NS": "Tata Consultancy Services", 
        "INFY.NS": "Infosys", 
        "HDFCBANK.NS": "HDFC Bank", 
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "State Bank of India", 
        "WIPRO.NS": "Wipro", 
        "HINDUNILVR.NS": "Hindustan Unilever",
        "ITC.NS": "ITC Limited", 
        "BAJFINANCE.NS": "Bajaj Finance", 
        "BHARTIARTL.NS": "Bharti Airtel",
        "LT.NS": "Larsen & Toubro", 
        "KOTAKBANK.NS": "Kotak Mahindra Bank", 
        "AXISBANK.NS": "Axis Bank",
        "ASIANPAINT.NS": "Asian Paints", 
        "HINDALCO.NS": "Hindalco Industries",
        "MARUTI.NS": "Maruti Suzuki", 
        "TITAN.NS": "Titan Company", 
        "SUNPHARMA.NS": "Sun Pharmaceutical",
        "NTPC.NS": "NTPC", 
        "ONGC.NS": "Oil and Natural Gas Corporation", 
        "POWERGRID.NS": "Power Grid Corporation",
        "M&M.NS": "Mahindra & Mahindra", 
        "ULTRACEMCO.NS": "UltraTech Cement", 
        "ADANIPORTS.NS": "Adani Ports",
        "ADANIENT.NS": "Adani Enterprises", 
        "HCLTECH.NS": "HCL Technologies", 
        "INDUSINDBK.NS": "IndusInd Bank"
    }
    
    company_name = company_map.get(ticker, ticker.split('.')[0])
    
    try:
        url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize={num_articles}&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"News API error: {data.get('message', 'Unknown error')}")
            return get_sample_news(ticker, num_articles)
            
        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "")
            })
        
        return articles
    except Exception as e:
        logger.error(f"News fetch error: {str(e)}")
        return get_sample_news(ticker, num_articles)

def get_sample_news(ticker, num_articles=5):
    """Generate sample news data when API is not available"""
    company_map = {
        "RELIANCE.NS": "Reliance Industries", 
        "TATAMOTORS.NS": "Tata Motors",
        "TCS.NS": "Tata Consultancy Services", 
        "INFY.NS": "Infosys", 
        "HDFCBANK.NS": "HDFC Bank", 
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "State Bank of India", 
        "WIPRO.NS": "Wipro", 
        "HINDUNILVR.NS": "Hindustan Unilever",
        "ITC.NS": "ITC Limited", 
        "BAJFINANCE.NS": "Bajaj Finance", 
        "BHARTIARTL.NS": "Bharti Airtel",
        "LT.NS": "Larsen & Toubro", 
        "KOTAKBANK.NS": "Kotak Mahindra Bank", 
        "AXISBANK.NS": "Axis Bank",
        "ASIANPAINT.NS": "Asian Paints", 
        "HINDALCO.NS": "Hindalco Industries",
        "MARUTI.NS": "Maruti Suzuki", 
        "TITAN.NS": "Titan Company", 
        "SUNPHARMA.NS": "Sun Pharmaceutical",
        "NTPC.NS": "NTPC", 
        "ONGC.NS": "Oil and Natural Gas Corporation", 
        "POWERGRID.NS": "Power Grid Corporation",
        "M&M.NS": "Mahindra & Mahindra", 
        "ULTRACEMCO.NS": "UltraTech Cement", 
        "ADANIPORTS.NS": "Adani Ports",
        "ADANIENT.NS": "Adani Enterprises", 
        "HCLTECH.NS": "HCL Technologies", 
        "INDUSINDBK.NS": "IndusInd Bank"
    }
    
    company_name = company_map.get(ticker, ticker.split('.')[0])
    
    sample_news = [
        {
            "title": f"{company_name} Reports Strong Quarterly Results",
            "description": f"{company_name} has reported better-than-expected quarterly results, with revenue growth of 15% year-over-year.",
            "url": "#",
            "publishedAt": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Financial Times"
        },
        {
            "title": f"Analysts Upgrade {company_name} to Buy Rating",
            "description": f"Several analysts have upgraded {company_name} to a buy rating, citing strong growth potential.",
            "url": "#",
            "publishedAt": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Bloomberg"
        },
        {
            "title": f"{company_name} Announces New Product Launch",
            "description": f"{company_name} has announced the launch of a new product that is expected to drive future growth.",
            "url": "#",
            "publishedAt": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Reuters"
        },
        {
            "title": f"{company_name} Expands into New Markets",
            "description": f"{company_name} is expanding into new international markets, which could significantly increase its addressable market.",
            "url": "#",
            "publishedAt": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Wall Street Journal"
        },
        {
            "title": f"Industry Trends Favor {company_name}",
            "description": f"Current industry trends are favoring companies like {company_name}, according to market analysts.",
            "url": "#",
            "publishedAt": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "CNBC"
        }
    ]
    
    return sample_news[:num_articles]

def analyze_sentiment(text):
    """Simple sentiment analysis function"""
    if not text:
        return "neutral"
    
    text = text.lower()
    positive_words = ["strong", "growth", "profit", "gain", "upgrade", "buy", "outperform", "beat", "positive"]
    negative_words = ["weak", "loss", "fall", "downgrade", "sell", "underperform", "miss", "negative", "drop"]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# ------------------ GROQ AI ASSISTANT ------------------
def get_groq_client():
    """Initialize Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def get_ai_response(client, prompt, context=""):
    """Get response from Groq AI assistant"""
    if not client:
        return "Groq API key not configured. Please add GROQ_API_KEY to your environment variables."
    
    try:
        system_prompt = f"""
        You are a financial analyst assistant specializing in Indian stocks. Provide insightful, data-driven analysis about stocks and investments.
        Use the following context information if relevant: {context}
        Be concise but informative in your responses.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return f"Error getting AI response: {str(e)}"

# ------------------ FEATURE ENGINEERING ------------------
def calculate_technical_indicators(data):
    """Calculate various technical indicators for stock data"""
    if data.empty or 'Close' not in data.columns:
        return data
    
    close_series = data['Close']
    
    # Check if we have enough data for calculations
    if len(data) < 20:
        st.warning(f"Insufficient data for technical indicators. Need at least 20 data points, got {len(data)}.")
        return data
    
    # Moving Averages
    try:
        data['SMA20'] = close_series.rolling(window=20).mean()
        if len(data) >= 50:
            data['SMA50'] = close_series.rolling(window=50).mean()
        data['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating moving averages: {str(e)}")
    
    # RSI - need at least 14 periods
    try:
        if len(data) >= 14:
            data['RSI'] = ta.momentum.rsi(close_series, window=14)
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
    
    # MACD - need at least 26 periods
    try:
        if len(data) >= 26:
            macd = ta.trend.MACD(close_series)
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Hist'] = macd.macd_diff()
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
    
    # Bollinger Bands - need at least 20 periods
    try:
        if len(data) >= 20:
            bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
            data['BB_Upper'] = bollinger.bollinger_hband()
            data['BB_Lower'] = bollinger.bollinger_lband()
            data['BB_Width'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
    
    # Volatility
    try:
        if len(data) >= 20:
            returns = close_series.pct_change()
            data['Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
    
    # Lagged returns
    try:
        for i in [1, 3, 5, 7]:
            if len(data) > i:
                data[f'Return_{i}d'] = close_series.pct_change(i)
    except Exception as e:
        logger.error(f"Error calculating lagged returns: {str(e)}")
    
    return data.dropna()

# ------------------ MODEL TRAINING ------------------
def prepare_features(data, forecast_horizon=30):
    """Prepare features for machine learning models"""
    if data.empty:
        return pd.DataFrame(), pd.Series()
    
    # Create target variable (future price change)
    data['Target'] = data['Close'].shift(-forecast_horizon) / data['Close'] - 1
    
    # Remove rows with NaN values
    data_clean = data.dropna()
    
    # Separate features and target
    feature_cols = [col for col in data_clean.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    X = data_clean[feature_cols]
    y = data_clean['Target']
    
    return X, y

def train_prophet_model(data, forecast_days):
    """Train Prophet model for time series forecasting"""
    if len(data) < 90:
        raise ValueError("Need at least 90 days of data for forecasting")
    
    # Prepare data for Prophet
    prophet_df = data[['Close']].reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Initialize and train model
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    
    model.fit(prophet_df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    return model, forecast

# ------------------ EVALUATION METRICS ------------------
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }

# ------------------ VISUALIZATION ------------------
def plot_stock_data(data, ticker):
    """Plot stock price data with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Moving averages
    if 'SMA20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA20'],
            mode='lines', name='20-day MA',
            line=dict(color='orange', width=2)
        ))
    
    if 'SMA50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA50'],
            mode='lines', name='50-day MA',
            line=dict(color='purple', width=2)
        ))
    
    fig.update_layout(
        title=f'{ticker} Price Movement',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_dark',
        height=500
    )
    
    return fig

def plot_technical_indicators(data):
    """Plot technical indicators"""
    fig = go.Figure()
    
    # Price and MACD
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        mode='lines', name='Close',
        line=dict(color='#4F8BF9')
    ))
    
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            mode='lines', name='MACD',
            line=dict(color='#FFA500'),
            yaxis='y2'
        ))
    
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            mode='lines', name='RSI',
            line=dict(color='#FF00FF'),
            yaxis='y3'
        ))
    
    fig.update_layout(
        title='Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='MACD',
            overlaying='y',
            side='right',
            position=0.85
        ),
        yaxis3=dict(
            title='RSI',
            overlaying='y',
            side='right',
            position=1.0,
            range=[0, 100]
        ),
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    # Add overbought/oversold lines for RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref="y3")
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref="y3")
    
    return fig

# ------------------ MAIN APPLICATION ------------------
def main():
    st.title("📈 Quantum Stock Prediction - India")
    st.markdown("A machine learning approach to Indian stock price prediction with news sentiment and AI assistant")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Indian stock selection
    indian_stocks = [
        "RELIANCE.NS", "TATAMOTORS.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
        "ICICIBANK.NS", "SBIN.NS", "WIPRO.NS", "HINDUNILVR.NS", "ITC.NS",
        "BAJFINANCE.NS", "BHARTIARTL.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "ASIANPAINT.NS", "HINDALCO.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS",
        "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "M&M.NS", "ULTRACEMCO.NS",
        "ADANIPORTS.NS", "ADANIENT.NS", "HCLTECH.NS", "INDUSINDBK.NS"
    ]
    
    ticker = st.sidebar.selectbox("Select Stock", indian_stocks, index=0)
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", start_date)
    with col2:
        end_date = st.date_input("End Date", end_date)
    
    # Forecast horizon
    forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
    
    # News articles count
    news_count = st.sidebar.slider("Number of News Articles", 3, 10, 5)
    
    # Fetch data
    with st.spinner('Fetching stock data...'):
        data = get_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        st.error("No data available for the selected ticker and date range.")
        
        # Provide helpful suggestions
        st.info("""
        **Troubleshooting tips:**
        1. Check if the ticker symbol is correct (e.g., RELIANCE.NS, TCS.NS)
        2. Ensure your date range includes trading days (avoid weekends and holidays)
        3. Try a shorter date range if the issue persists
        """)
        
        # Show sample data for demonstration
        if st.checkbox("Show sample data for demonstration"):
            sample_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            sample_data = pd.DataFrame({
                'Open': np.random.uniform(100, 200, len(sample_dates)),
                'High': np.random.uniform(200, 250, len(sample_dates)),
                'Low': np.random.uniform(80, 100, len(sample_dates)),
                'Close': np.random.uniform(100, 200, len(sample_dates)),
                'Volume': np.random.uniform(1000000, 5000000, len(sample_dates))
            }, index=sample_dates)
            st.dataframe(sample_data)
        return
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Technical Analysis", "Prediction", "News Sentiment", "AI Assistant"])
    
    # Data Overview Tab
    with tab1:
        st.header("Data Overview")
        
        # Display basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Ensure we're working with scalar values
        current_price = float(data['Close'].iloc[-1]) if not data.empty else 0
        prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        
        daily_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        
        # Get volatility and RSI values safely
        volatility_value = float(data['Volatility'].iloc[-1]) if 'Volatility' in data.columns and not data['Volatility'].empty else 0
        rsi_value = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns and not data['RSI'].empty else 0
        
        col1.metric("Current Price", f"₹{current_price:.2f}")
        col2.metric("Daily Change", f"{daily_change:.2f}%")
        col3.metric("30-Day Volatility", f"{volatility_value*100:.2f}%" if volatility_value else "N/A")
        col4.metric("RSI", f"{rsi_value:.2f}" if rsi_value else "N/A")
        
        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(data.tail(10))
        
        # Display price chart
        st.subheader("Price Chart")
        st.plotly_chart(plot_stock_data(data, ticker), use_container_width=True)
    
    # Technical Analysis Tab
    with tab2:
        st.header("Technical Analysis")
    
        # Display technical indicators
        st.subheader("Technical Indicators")
        st.plotly_chart(plot_technical_indicators(data), use_container_width=True)
    
        # Feature importance (simplified)
        st.subheader("Feature Correlation with Price")
        if len(data) > 30:  # Ensure we have enough data
            numeric_data = data.select_dtypes(include=[np.number])
        
            # Check if 'Close' exists in numeric_data
            if 'Close' in numeric_data.columns:
                corr = numeric_data.corr()['Close'].sort_values(ascending=False)
            
                # Remove price itself from correlation list
                corr = corr.drop('Close', errors='ignore')
            
                fig = px.bar(
                    x=corr.values, 
                    y=corr.index, 
                    orientation='h',
                    title='Feature Correlation with Closing Price',
                    labels={'x': 'Correlation', 'y': 'Feature'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Close price data not available for correlation analysis")
    
    # Prediction Tab
    with tab3:
        st.header("Price Prediction")
        
        # Model selection
        model_type = st.radio("Select Model", ["Prophet (Time Series)", "Linear Model (Technical Indicators)"])
        
        if model_type == "Prophet (Time Series)":
            try:
                with st.spinner('Training Prophet model...'):
                    model, forecast = train_prophet_model(data, forecast_days)
                
                # Display forecast
                st.subheader("Prophet Forecast")
                fig = model.plot(forecast)
                st.pyplot(fig)
                
                # Display components
                st.subheader("Forecast Components")
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
                
            except Exception as e:
                st.error(f"Error in Prophet forecasting: {str(e)}")
        
        else:  # Linear Model
            with st.spinner('Preparing features and training model...'):
                # Prepare features
                X, y = prepare_features(data, forecast_horizon=forecast_days)
                
                if X.empty:
                    st.warning("Not enough data to train the model. Try selecting a longer time period.")
                else:
                    # Train-test split (80-20)
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Train simple linear model
                    from sklearn.linear_model import LinearRegression
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression())
                    ])
                    
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_test, y_pred)
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    col2.metric("MAE", f"{metrics['MAE']:.4f}")
                    col3.metric("MSE", f"{metrics['MSE']:.4f}")
                    
                    # Plot predictions vs actual
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(y_test)),
                        y=y_test.values,
                        mode='lines',
                        name='Actual'
                    ))
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(y_pred)),
                        y=y_pred,
                        mode='lines',
                        name='Predicted'
                    ))
                    fig.update_layout(
                        title='Actual vs Predicted Price Changes',
                        xaxis_title='Time',
                        yaxis_title='Price Change',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    if hasattr(model.named_steps['regressor'], 'coef_'):
                        importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.named_steps['regressor'].coef_
                        }).sort_values('importance', key=abs, ascending=False)
                        
                        fig = px.bar(
                            importance, 
                            x='importance', 
                            y='feature', 
                            orientation='h',
                            title='Feature Importance (Linear Model)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # News Sentiment Tab
    with tab4:
        st.header("News Sentiment Analysis")
        
        with st.spinner('Fetching news articles...'):
            news_articles = get_stock_news(ticker, news_count)
        
        if not news_articles:
            st.warning("No news articles found for this stock.")
        else:
            # Calculate overall sentiment
            sentiments = []
            for article in news_articles:
                title_sentiment = analyze_sentiment(article['title'])
                desc_sentiment = analyze_sentiment(article.get('description', ''))
                # Weight title more heavily than description
                overall_sentiment = title_sentiment if title_sentiment != 'neutral' else desc_sentiment
                sentiments.append(overall_sentiment)
            
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            neutral_count = sentiments.count('neutral')
            
            # Display sentiment summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive Articles", positive_count)
            col2.metric("Negative Articles", negative_count)
            col3.metric("Neutral Articles", neutral_count)
            
            # Sentiment gauge
            if positive_count + negative_count > 0:
                sentiment_score = positive_count / (positive_count + negative_count) * 100
            else:
                sentiment_score = 50  # Neutral if no positive/negative articles
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_score,
                title = {'text': "Overall Sentiment Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display news articles
            st.subheader("Recent News Articles")
            for i, article in enumerate(news_articles):
                sentiment = sentiments[i]
                sentiment_color = {
                    "positive": "🟢",
                    "negative": "🔴",
                    "neutral": "🟡"
                }[sentiment]
                
                with st.expander(f"{sentiment_color} {article['title']}"):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Published:** {article['publishedAt'][:10]}")
                    st.write(f"**Description:** {article.get('description', 'No description available.')}")
                    if article['url'] != "#":
                        st.write(f"[Read more]({article['url']})")
    
    # AI Assistant Tab
    with tab5:
        st.header("AI Investment Assistant")
        
        # Initialize Groq client
        groq_client = get_groq_client()
        
        if not groq_client:
            st.warning("Groq API key not configured. Please add GROQ_API_KEY to your environment variables to use the AI assistant.")
        
        # Create context from stock data
        if not data.empty:
            current_price = float(data['Close'].iloc[-1]) if not data.empty else 0
            price_change = ((current_price - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0]) * 100) if len(data) > 1 else 0
            
            context = f"""
            Stock: {ticker}
            Current Price: ₹{current_price:.2f}
            Overall Change: {price_change:.2f}%
            """
            
            if 'RSI' in data.columns and not data['RSI'].empty:
                context += f"RSI: {float(data['RSI'].iloc[-1]):.2f}\n"
            if 'Volatility' in data.columns and not data['Volatility'].empty:
                context += f"Volatility: {float(data['Volatility'].iloc[-1])*100:.2f}%\n"
        else:
            context = f"Stock: {ticker}"
        
        # Sample questions
        st.subheader("Sample Questions")
        col1, col2, col3 = st.columns(3)
        
        sample_questions = [
            "What is the investment outlook for this stock?",
            "What are the key risks associated with this stock?",
            "How does technical analysis look for this stock?",
            "What factors could drive this stock's price higher?",
            "Should I consider this stock for long-term investment?",
            "What is the analyst consensus for this stock?"
        ]
        
        if col1.button(sample_questions[0]):
            st.session_state.user_question = sample_questions[0]
        if col1.button(sample_questions[1]):
            st.session_state.user_question = sample_questions[1]
        if col2.button(sample_questions[2]):
            st.session_state.user_question = sample_questions[2]
        if col2.button(sample_questions[3]):
            st.session_state.user_question = sample_questions[3]
        if col3.button(sample_questions[4]):
            st.session_state.user_question = sample_questions[4]
        if col3.button(sample_questions[5]):
            st.session_state.user_question = sample_questions[5]
        
        # User input
        user_question = st.text_input(
            "Ask a question about this stock:",
            value=st.session_state.get('user_question', ''),
            key="user_input"
        )
        
        if st.button("Get Answer") and user_question:
            with st.spinner('Thinking...'):
                response = get_ai_response(groq_client, user_question, context)
                
                st.subheader("AI Assistant Response")
                st.info(response)

if __name__ == "__main__":
    main()
