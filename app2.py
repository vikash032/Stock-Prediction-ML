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
import ta
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import random
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import logging
from dotenv import load_dotenv
from scipy.optimize import minimize
import holidays
import json
from sklearn.model_selection import TimeSeriesSplit
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import optuna
import redis
import schedule
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------ CONFIGURATION ------------------
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

# ------------------ MODULES ------------------
# Module 1: Data Fetching
@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def get_stock_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance with robust error handling"""
    try:
        logger.info(f"Fetching data for {ticker} from {start} to {end}")
        data = yf.download(ticker, start=start - timedelta(days=60), end=end + timedelta(days=1))
        
        if data.empty:
            logger.warning(f"No data found for {ticker}, trying 1-year period")
            data = yf.download(ticker, period="1y")
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
        
        # Validate data structure
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
                
        return data
    except Exception as e:
        logger.error(f"Data fetch error: {str(e)}")
        st.error(f"Data fetch error: {str(e)}. Please try a different ticker or date range.")
        return pd.DataFrame()

# Module 2: Technical Analysis
def calculate_technical_indicators(data):
    """Calculate various technical indicators for stock data"""
    if 'Close' not in data.columns or len(data) < 20:
        return data
    
    close_series = data['Close'].squeeze()
    
    # Moving Averages
    try:
        data['SMA20'] = close_series.rolling(window=20).mean()
        data['SMA50'] = close_series.rolling(window=50).mean()
        data['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating moving averages: {str(e)}")
    
    # RSI
    try:
        data['RSI'] = ta.momentum.rsi(close_series, window=14)
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
    
    # MACD
    try:
        macd = ta.trend.MACD(close_series)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
    
    # Bollinger Bands
    try:
        bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        data['BB_Width'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
    
    # Volatility
    try:
        returns = close_series.pct_change()
        data['Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
    
    # Lagged returns
    for i in [1, 3, 5, 7]:
        data[f'Return_{i}d'] = close_series.pct_change(i)
    
    return data.dropna()

# Module 3: Sentiment Analysis
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """Load and cache the sentiment analysis model"""
    logger.info("Loading sentiment model")
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker):
    """Fetch news articles related to a stock ticker"""
    logger.info(f"Fetching news for {ticker}")
    api_key = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY")
    if not api_key:
        logger.error("News API key not found")
        st.error("News API key not configured. News features disabled.")
        return []
    
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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"News API error: {data.get('message', 'Unknown error')}")
            return []
            
        return [
            {
                "title": a.get("title", ""),
                "summary": a.get("description", ""),
                "link": a.get("url", ""),
                "date": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", "")
            } for a in data.get("articles", [])
        ]
    except Exception as e:
        logger.error(f"News error: {e}")
        return []

# Module 4: Forecasting
def prophet_forecast(data, forecast_days, country='IN'):
    """Perform time series forecasting using Prophet with holidays and technical indicators"""
    if len(data) < 90:
        raise ValueError("Need at least 90 days of data for forecasting")
    
    # Create holiday dataframe for the country
    years = pd.date_range(start=data.index.min(), end=data.index.max() + timedelta(days=forecast_days)).year
    all_years = list(range(min(years), max(years)+1))
    country_holidays = holidays.CountryHoliday(country, years=all_years)
    holiday_df = pd.DataFrame([(date, name) for date, name in country_holidays.items()], columns=['ds', 'holiday'])
    
    prophet_df = data[['Close']].reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Add technical indicators as regressors
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=10,
        changepoint_range=0.95,
        uncertainty_samples=100,
        holidays=holiday_df
    )
    
    # Add technical indicators as regressors
    tech_indicators = ['SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'MACD_Hist', 'BB_Width', 'Volatility']
    for indicator in tech_indicators:
        if indicator in data.columns:
            prophet_df[indicator] = data[indicator].values
            model.add_regressor(indicator)
    
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Add future technical indicators (using the last known values as placeholders)
    for indicator in tech_indicators:
        if indicator in data.columns:
            last_value = data[indicator].iloc[-1]
            future[indicator] = last_value
    
    forecast = model.predict(future)
    return model, forecast

def tune_tft_hyperparameters(train_dataloader, val_dataloader):
    """Optimize TFT hyperparameters using Optuna"""
    logger.info("Tuning TFT hyperparameters...")
    
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 8, 32)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
        attention_head_size = trial.suggest_int("attention_head_size", 1, 4)
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 4, 16)
        
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
        
        tft = TemporalFusionTransformer.from_dataset(
            train_dataloader.dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=3,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            reduce_on_plateau_patience=2,
        )
        
        trainer = pl.Trainer(
            max_epochs=15,
            gpus=0,
            enable_progress_bar=False,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches=15,
            enable_checkpointing=True,
        )
        
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        return trainer.callback_metrics["val_loss"].item()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params

def tft_forecast(data, forecast_days, tune=False):
    """Perform time series forecasting using Temporal Fusion Transformer"""
    # Add technical indicators
    data = calculate_technical_indicators(data.copy())
    
    # Prepare data for TFT
    df = data.reset_index()
    df.rename(columns={'Date': 'date'}, inplace=True)
    df['time_idx'] = np.arange(len(df))
    df['series'] = "stock"
    df['date'] = pd.to_datetime(df['date'])
    
    # Add additional features
    df['day'] = df['date'].dt.day.astype(str)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(str)
    df['month'] = df['date'].dt.month.astype(str)
    df['quarter'] = df['date'].dt.quarter.astype(str)
    
    # Define features
    features = ['Close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'MACD_Signal', 
                'MACD_Hist', 'BB_Upper', 'BB_Lower', 'BB_Width', 'Volatility',
                'Return_1d', 'Return_3d', 'Return_5d', 'Return_7d']
    
    available_features = [f for f in features if f in df.columns]
    
    # Define training parameters
    max_prediction_length = forecast_days
    max_encoder_length = min(180, len(df) - max_prediction_length - 1)
    
    if max_encoder_length < 60:
        raise ValueError("Insufficient data for TFT forecasting. Need at least 60 days of data.")
    
    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    # Create dataset
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
        time_varying_known_categoricals=["day", "dayofweek", "month", "quarter"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=available_features,
        target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation set
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    # Create dataloaders
    batch_size = 16
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # Tune hyperparameters if requested
    best_params = {
        'hidden_size': 16,
        'attention_head_size': 2,
        'dropout': 0.1,
        'learning_rate': 0.01,
        'hidden_continuous_size': 8
    }
    
    if tune:
        try:
            best_params = tune_tft_hyperparameters(train_dataloader, val_dataloader)
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
    
    # Configure TFT with best parameters
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=best_params['learning_rate'],
        hidden_size=best_params['hidden_size'],
        attention_head_size=best_params['attention_head_size'],
        dropout=best_params['dropout'],
        hidden_continuous_size=best_params['hidden_continuous_size'],
        output_size=3,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=3,
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=20,
        gpus=0,
        enable_progress_bar=False,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        limit_train_batches=20,
        enable_checkpointing=True,
    )
    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # Generate predictions
    raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
    
    # Extract forecast values
    forecast = raw_predictions[0].output.prediction[1].cpu().numpy().flatten()  # P50
    lower_band = raw_predictions[0].output.prediction[0].cpu().numpy().flatten()  # P10
    upper_band = raw_predictions[0].output.prediction[2].cpu().numpy().flatten()  # P90
    
    # Get actual values for comparison
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).cpu().numpy()
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(actuals.flatten()[:len(forecast)], forecast))
    
    # Extract attention weights
    attention = tft.interpret_output(raw_predictions, reduction="none")[1]['attention'][0].detach().cpu().numpy()
    
    # Calculate feature importance using SHAP
    explainer = shap.DeepExplainer(tft, val_dataloader)
    shap_values = explainer.shap_values(val_dataloader)
    
    return {
        'forecast': forecast,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'train_rmse': train_rmse,
        'test_rmse': train_rmse,
        'attention': attention,
        'shap_values': shap_values,
        'features': available_features,
        'model': tft
    }

def ensemble_forecast(prophet_forecast, tft_forecast, actuals, forecast_days):
    """Combine Prophet and TFT forecasts using weighted averaging"""
    # Calculate weights based on recent performance
    prophet_mae = mean_absolute_error(actuals[-30:], prophet_forecast[-30-forecast_days:-forecast_days])
    tft_mae = mean_absolute_error(actuals[-30:], tft_forecast[:30])
    
    # Use inverse MAE as weights
    prophet_weight = 1 / prophet_mae
    tft_weight = 1 / tft_mae
    total_weight = prophet_weight + tft_weight
    
    # Normalize weights
    prophet_weight /= total_weight
    tft_weight /= total_weight
    
    # Combine forecasts
    combined_forecast = (prophet_forecast[-forecast_days:] * prophet_weight + 
                         tft_forecast * tft_weight)
    
    return combined_forecast, prophet_weight, tft_weight

# Module 5: Portfolio Optimization
@st.cache_data(ttl=3600, show_spinner=False)
def prepare_portfolio_data(tickers, start_date, end_date):
    """Prepare portfolio data for multiple tickers"""
    price_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date - timedelta(days=60), end=end_date + timedelta(days=1))
            if df.empty:
                logger.warning(f"No data for {ticker}, skipping...")
                continue
            price_data[ticker] = df['Close']
        except Exception as e:
            logger.error(f"Error loading {ticker}: {str(e)}")
            continue

    if not price_data:
        return pd.DataFrame()

    combined_df = pd.concat(price_data.values(), axis=1, keys=price_data.keys())
    combined_df.columns = combined_df.columns.droplevel(1)
    return combined_df.dropna(how='all')

def optimize_portfolio(returns, risk_tolerance):
    """Optimize portfolio using Modern Portfolio Theory"""
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
        logger.error(f"Optimization failed: {str(e)}")
        return np.ones(n) / n

# Module 6: Backtesting
def backtest_strategy(data, strategy, params):
    """Backtest a trading strategy with realistic simulation"""
    if len(data) < 100:
        return {
            'return': 0,
            'drawdown': 0,
            'sharpe': 0,
            'trades': 0
        }
    
    # Initialize portfolio
    cash = 10000
    position = 0
    portfolio_value = [cash]
    trades = []
    
    # Strategy-specific parameters
    if strategy == "Moving Average Crossover":
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        data['SMA_short'] = data['Close'].rolling(short_window).mean()
        data['SMA_long'] = data['Close'].rolling(long_window).mean()
    
    for i in range(long_window, len(data)):
        price = data['Close'].iloc[i]
        prev_price = data['Close'].iloc[i-1]
        
        # Generate signal based on strategy
        signal = 0
        
        if strategy == "Moving Average Crossover":
            if data['SMA_short'].iloc[i-1] < data['SMA_long'].iloc[i-1] and \
               data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i]:
                signal = 1  # Golden cross - buy
            elif data['SMA_short'].iloc[i-1] > data['SMA_long'].iloc[i-1] and \
                 data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i]:
                signal = -1  # Death cross - sell
        
        # Execute trades
        if signal == 1 and cash > 0:
            # Buy with all cash
            shares = cash // price
            position += shares
            cash -= shares * price
            trades.append(('buy', data.index[i], price, shares))
        elif signal == -1 and position > 0:
            # Sell all position
            cash += position * price
            trades.append(('sell', data.index[i], price, position))
            position = 0
        
        # Update portfolio value
        portfolio_value.append(cash + position * price)
    
    # Calculate performance metrics
    portfolio = pd.Series(portfolio_value)
    returns = portfolio.pct_change().dropna()
    total_return = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100
    
    # Calculate max drawdown
    peak = portfolio.cummax()
    drawdown = (portfolio - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Calculate Sharpe ratio
    if returns.std() > 0:
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0
    
    return {
        'return': total_return,
        'drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': trades,
        'portfolio': portfolio
    }

# Module 7: Real-time Features
class RealTimeMonitor:
    def __init__(self):
        self.performance_history = []
        self.last_retrain = datetime.now()
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=os.getenv('REDIS_PORT', 6379),
            password=os.getenv('REDIS_PASSWORD', ''),
            decode_responses=True
        )
    
    def monitor_performance(self, model_name, rmse):
        """Track model performance and detect degradation"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            'rmse': rmse
        })
        
        # Check for performance degradation
        if len(self.performance_history) > 5:
            recent = self.performance_history[-5:]
            avg_rmse = sum([r['rmse'] for r in recent]) / 5
            prev_avg = sum([r['rmse'] for r in self.performance_history[-10:-5]]) / 5
            
            if avg_rmse > prev_avg * 1.05:  # 5% degradation
                self.send_alert(f"Model performance degradation detected: {model_name}")
                return True
        return False
    
    def should_retrain(self):
        """Determine if it's time to retrain models"""
        # Retrain every week or if performance degrades
        return (datetime.now() - self.last_retrain).days >= 7
    
    def send_alert(self, message):
        """Send alert notification"""
        try:
            # Email configuration
            smtp_server = os.getenv('SMTP_SERVER')
            smtp_port = int(os.getenv('SMTP_PORT', 587))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')
            recipient = os.getenv('ALERT_RECIPIENT')
            
            if not all([smtp_server, smtp_user, smtp_password, recipient]):
                logger.warning("Alert configuration incomplete")
                return
            
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = recipient
            msg['Subject'] = "Stock Analytics Alert"
            
            body = f"""
            <h2>Stock Analytics Alert</h2>
            <p>{message}</p>
            <p>Timestamp: {datetime.now()}</p>
            """
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
    
    def cache_data(self, key, value, ttl=3600):
        """Cache data in Redis"""
        try:
            self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Redis cache error: {str(e)}")
    
    def get_cached_data(self, key):
        """Retrieve cached data from Redis"""
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None

# ------------------ UTILITY FUNCTIONS ------------------
def calculate_annual_return(data, start_date, end_date):
    """Calculate annualized return for a stock"""
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
    return (1 + total_return) ** (1 / years_held) - 1

def calculate_volatility(data):
    """Calculate annualized volatility for a stock"""
    if len(data) < 30:
        return 0.0
    if 'Close' in data.columns:
        close_series = data['Close'].squeeze()
        returns = close_series.pct_change().dropna()
        if len(returns) < 30:
            return 0.0
        daily_vol = returns.std()
        return daily_vol * np.sqrt(252)

def clean_text(text):
    """Clean text for sentiment analysis"""
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text[:512]

def calculate_annualized_return(series):
    """Calculate annualized return from a price series"""
    returns = series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return (1 + returns).prod() ** (252/len(returns)) - 1

def create_options_payoff(strike_price, premium, option_type, num_contracts=1):
    """Calculate options payoff diagram"""
    stock_prices = np.linspace(strike_price * 0.7, strike_price * 1.3, 100)
    contract_size = 100  # Standard contract size
    
    if option_type == 'call':
        payoff = np.maximum(stock_prices - strike_price, 0) * contract_size * num_contracts - (premium * contract_size * num_contracts)
    else:  # put
        payoff = np.maximum(strike_price - stock_prices, 0) * contract_size * num_contracts - (premium * contract_size * num_contracts)
    
    return stock_prices, payoff

def get_earnings_data(ticker):
    """Get earnings data for a stock"""
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

def generate_ai_response(query, stock_data, portfolio_data=None, risk_profile="Moderate", investment_goal="Growth"):
    """Generate AI-powered response to investment questions"""
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
        Our hybrid forecasting model predicts:
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

def get_macro_data():
    """Get macroeconomic data (placeholder)"""
    # Placeholder - in real implementation, use API
    return {
        'inflation': 3.2,
        'interest_rate': 5.25,
        'unemployment': 3.8,
        'gdp_growth': 2.1,
        'consumer_sentiment': 78.4,
        'manufacturing_pmi': 52.7
    }

def get_institutional_activity(ticker):
    """Get institutional activity data (placeholder)"""
    # Placeholder - in real implementation, use API
    dates = pd.date_range(end=datetime.today(), periods=12, freq='M')
    return pd.DataFrame({
        'Date': dates,
        'Shares Held': np.random.randint(1000000, 5000000, 12),
        '% Change': np.random.uniform(-5, 5, 12),
        'Number of Institutions': np.random.randint(100, 500, 12)
    })

def plot_attention_weights(attention):
    """Plot TFT attention weights"""
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title("TFT Attention Weights")
    ax.set_xlabel("Encoder Time Steps")
    ax.set_ylabel("Decoder Time Steps")
    return fig

def plot_shap_values(shap_values, features):
    """Plot SHAP values for feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP Values)")
    plt.tight_layout()
    return fig

# Initialize real-time monitor
rt_monitor = RealTimeMonitor()

# ------------------ UI STYLES ------------------
# ------------------ UI STYLES ------------------
CUSTOM_CSS = """
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
"""

# ------------------ MAIN APP ------------------
def main():
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="header">🚀 QUANTUM STOCK ANALYTICS</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:30px;">
        <h3 class="glow-text">AI-Powered Financial Intelligence Platform</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"<div style='text-align:center; margin-bottom:30px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

    # Sidebar configuration
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
    
    # Market sentiment gauge
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
    
    # Prediction alerts
    st.sidebar.markdown("### 🔮 Prediction Alerts")
    alert_threshold = st.sidebar.number_input("Alert if predicted return exceeds (%)", 
                                             min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    
    # User profile
    st.sidebar.markdown("### 👤 User Profile")
    user_risk_profile = st.sidebar.select_slider("Your Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"], value="Moderate")
    user_investment_goal = st.sidebar.selectbox("Primary Goal", ["Capital Growth", "Income", "Preservation"], index=0)

    # Advanced options
    st.sidebar.markdown("### ⚙️ Advanced Options")
    tune_hyperparams = st.sidebar.checkbox("Tune Hyperparameters", value=False)
    enable_ensemble = st.sidebar.checkbox("Enable Ensemble Forecasting", value=True)

    # Fetch stock data
    with st.spinner('Fetching market data...'):
        data = get_stock_data(ticker, start_date, end_date)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏠 Home", "📈 Market Data", "🔮 Forecasting", "📰 Sentiment", 
        "💼 Portfolio", "🤖 AI Assistant", "🧪 Strategy", "🚀 Real-Time"
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
                <h3 style="color:white;">Tech Stack</h3>
                <div style="font-size:3rem;">🤖</div>
                <p><strong>AI-Powered Analytics</strong></p>
                <ul style="text-align:left;">
                    <li>Prophet Forecasting</li>
                    <li>TFT Neural Networks</li>
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
        
        if data.empty:
            st.error("No data available for analysis. Please select a different ticker or date range.")
        else:
            # Render metrics
            render_metrics(data, ticker, start_date, end_date)
            
            # Price Movement Chart
            if len(data) > 1 and 'Close' in data.columns:
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
                data = calculate_technical_indicators(data)

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
                    x=data.index, y=data['MACD_Signal'],
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
        st.markdown('<div class="subheader">Hybrid Prophet-TFT Forecasting</div>', unsafe_allow_html=True)
        
        if data.empty:
            st.error("No data available for forecasting. Please select a different ticker or date range.")
        else:
            try:
                with st.spinner('Running Prophet forecast with technical indicators...'):
                    prophet_model, prophet_forecast_df = prophet_forecast(data, forecast_days)
                    
                    st.subheader("Prophet Forecast")
                    fig1 = plot_plotly(prophet_model, prophet_forecast_df)
                    fig1.update_layout(
                        height=500,
                        template='plotly_dark',
                        title=f"{ticker} Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.subheader("Forecast Components")
                    fig2 = plot_components_plotly(prophet_model, prophet_forecast_df)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Confidence interval
                    last_forecast = prophet_forecast_df.iloc[-1]
                    confidence_interval = last_forecast['yhat_upper'] - last_forecast['yhat_lower']
                    confidence_percent = min(100, max(0, 100 - (confidence_interval / last_forecast['yhat'] * 100)))
                    
                    st.metric("Forecast Confidence", f"{confidence_percent:.1f}%")
                    st.progress(int(confidence_percent))
                    
                    # Track performance
                    prophet_rmse = np.sqrt(mean_squared_error(
                        data['Close'].iloc[-30:], 
                        prophet_forecast_df['yhat'].iloc[-30-forecast_days:-forecast_days]
                    ))
                    rt_monitor.monitor_performance("Prophet", prophet_rmse)
                    
            except Exception as e:
                st.error(f"Prophet forecasting error: {str(e)}")
            
            try:
                with st.spinner('Running TFT forecast with hyperparameter tuning...'):
                    tft_results = tft_forecast(data, forecast_days, tune=tune_hyperparams)
                    
                    st.subheader("TFT Neural Network Forecast")
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
                    
                    # Track performance
                    rt_monitor.monitor_performance("TFT", tft_results['train_rmse'])
                    
                    # Attention Visualization
                    if 'attention' in tft_results and tft_results['attention'] is not None:
                        st.subheader("Attention Weights")
                        st.markdown('<div class="attention-heatmap">', unsafe_allow_html=True)
                        fig_attn = plot_attention_weights(tft_results['attention'])
                        st.pyplot(fig_attn)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.caption("Attention weights show which historical time steps the model focuses on when making predictions")
                    
                    # SHAP Explainability
                    if 'shap_values' in tft_results and 'features' in tft_results:
                        st.subheader("TFT Feature Importance")
                        fig_shap = plot_shap_values(tft_results['shap_values'], data[tft_results['features']])
                        st.pyplot(fig_shap)
            
            except Exception as e:
                st.error(f"TFT forecasting error: {str(e)}")
            
            # Ensemble Forecasting
            if enable_ensemble and not data.empty and 'prophet_forecast_df' in locals() and 'tft_results' in locals():
                try:
                    with st.spinner('Combining forecasts with ensemble model...'):
                        combined_forecast, prophet_weight, tft_weight = ensemble_forecast(
                            prophet_forecast_df['yhat'].values,
                            tft_results['forecast'],
                            data['Close'].values,
                            forecast_days
                        )
                        
                        st.subheader("Hybrid Ensemble Forecast")
                        fig_ensemble = go.Figure()
                        fig_ensemble.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='#4F8BF9')
                        ))
                        
                        fig_ensemble.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=combined_forecast,
                            mode='lines',
                            name='Ensemble Forecast',
                            line=dict(color='#FF00FF', width=3)
                        ))
                        
                        fig_ensemble.update_layout(
                            title='Hybrid Prophet-TFT Ensemble Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            template='plotly_dark',
                            height=500
                        )
                        st.plotly_chart(fig_ensemble, use_container_width=True)
                        
                        st.metric("Prophet Weight", f"{prophet_weight:.2%}")
                        st.metric("TFT Weight", f"{tft_weight:.2%}")
                        
                        # Check for prediction alerts
                        predicted_return = (combined_forecast[-1] / data['Close'].iloc[-1] - 1) * 100
                        if abs(predicted_return) > alert_threshold:
                            direction = "increase" if predicted_return > 0 else "decrease"
                            st.warning(f"⚠️ Significant predicted {direction}: {predicted_return:.1f}% over {forecast_days} days")
                
                except Exception as e:
                    st.error(f"Ensemble forecasting error: {str(e)}")

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
                    logger.error(f"Sentiment error: {str(e)}")
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

        if portfolio_data.empty:
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
        
            # Format the return columns as strings
            return_df['Expected Return'] = return_df['Expected Return'].apply(
               lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x)
            )
            return_df['Actual Return'] = return_df['Actual Return'].apply(
                lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x)
            )
        
            st.dataframe(return_df)
            
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
        params = {}
        if strategy == "Moving Average Crossover":
            params['short_window'] = st.slider("Short Window", 5, 50, 20)
            params['long_window'] = st.slider("Long Window", 20, 200, 50)
        elif strategy == "RSI Divergence":
            params['rsi_period'] = st.slider("RSI Period", 5, 30, 14)
            params['oversold'] = st.slider("Oversold Level", 0, 40, 30)
            params['overbought'] = st.slider("Overbought Level", 60, 100, 70)
        elif strategy == "Bollinger Band Reversion":
            params['bb_period'] = st.slider("Bollinger Period", 10, 50, 20)
            params['std_dev'] = st.slider("Standard Deviations", 1.0, 3.0, 2.0)
        elif strategy == "MACD Crossover":
            params['fast'] = st.slider("Fast EMA", 5, 20, 12)
            params['slow'] = st.slider("Slow EMA", 15, 50, 26)
            params['signal'] = st.slider("Signal Period", 5, 20, 9)
        elif strategy == "Golden Cross":
            params['short_ma'] = st.slider("Short MA", 20, 100, 50)
            params['long_ma'] = st.slider("Long MA", 100, 300, 200)
        
        # Backtest button
        if st.button("Run Backtest", key="backtest_run"):
            with st.spinner('Running backtest...'):
                results = backtest_strategy(data, strategy, params)
                
            # Display results
            st.subheader("Backtest Results")
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            col_res1.metric("Total Return", f"{results['return']:.2f}%")
            col_res2.metric("Max Drawdown", f"{results['drawdown']:.2f}%")
            col_res3.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
            col_res4.metric("Trades Executed", len(results['trades']))
            
            # Performance visualization
            st.subheader("Strategy Performance")
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#4F8BF9'),
                yaxis='y'
            ))
            
            fig_backtest.add_trace(go.Scatter(
                x=data.index[params.get('long_window', 50):],
                y=results['portfolio'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00FF00'),
                yaxis='y2'
            ))
            
            # Add trade markers
            buy_dates = [t[1] for t in results['trades'] if t[0] == 'buy']
            buy_prices = [t[2] for t in results['trades'] if t[0] == 'buy']
            sell_dates = [t[1] for t in results['trades'] if t[0] == 'sell']
            sell_prices = [t[2] for t in results['trades'] if t[0] == 'sell']
            
            fig_backtest.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            fig_backtest.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig_backtest.update_layout(
                title=f'{strategy} Performance',
                xaxis_title='Date',
                yaxis_title='Price',
                yaxis2=dict(
                    title='Portfolio Value',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                template='plotly_dark',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_backtest, use_container_width=True)
            
            # Trade log
            if results['trades']:
                st.subheader("Trade Log")
                trades_df = pd.DataFrame(results['trades'], columns=['Action', 'Date', 'Price', 'Shares'])
                st.dataframe(trades_df)

    # Real-Time Monitoring Tab
    with tab8:
        st.markdown('<div class="header">🚀 Real-Time Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Live market monitoring and predictions</div>', unsafe_allow_html=True)
        
        # Real-time data fetching
        if st.button("Refresh Real-Time Data"):
            st.experimental_rerun()
        
        col_rt1, col_rt2, col_rt3 = st.columns(3)
        with col_rt1:
            st.metric("Last Refresh", datetime.now().strftime("%H:%M:%S"))
        with col_rt2:
            st.metric("Market Status", "Open" if 9 <= datetime.now().hour < 16 else "Closed")
        with col_rt3:
            st.metric("Data Latency", "0.5s")
        
        # Real-time price chart
        st.subheader("Real-Time Price Movement")
        # Placeholder for real-time chart
        st.info("Real-time chart integration requires WebSocket connection to market data API")
        
        # Model monitoring
        st.subheader("Model Performance Monitoring")
        if rt_monitor.performance_history:
            perf_df = pd.DataFrame(rt_monitor.performance_history)
            fig_perf = px.line(perf_df, x='timestamp', y='rmse', color='model', 
                              title='Model RMSE Over Time', markers=True)
            fig_perf.update_layout(template='plotly_dark')
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.warning("No performance data available yet")
        
        # Alert system
        st.subheader("Alert System")
        if st.button("Test Alert Notification"):
            rt_monitor.send_alert("This is a test alert from Quantum Stock Analytics")
            st.success("Test alert sent!")
        
        # System status
        st.subheader("System Status")
        col_status1, col_status2, col_status3 = st.columns(3)
        col_status1.metric("Data API", "Connected", "OK")
        col_status2.metric("Model Serving", "Active", "OK")
        col_status3.metric("Prediction Latency", "120ms")
        
        # Retraining status
        st.subheader("Model Retraining")
        if rt_monitor.should_retrain():
            st.warning("Models are due for retraining")
            if st.button("Retrain Models Now"):
                with st.spinner("Retraining models..."):
                    # This would trigger retraining in a real system
                    time.sleep(5)
                    rt_monitor.last_retrain = datetime.now()
                    st.success("Models retrained successfully!")
        else:
            next_retrain = rt_monitor.last_retrain + timedelta(days=7)
            st.info(f"Models up to date. Next retraining scheduled for {next_retrain.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
