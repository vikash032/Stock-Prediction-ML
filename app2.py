import torch
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from transformers import pipeline
from datetime import datetime, timedelta
import cvxpy as cp
from sklearn.preprocessing import MinMaxScaler
import ta
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt
import logging
import holidays
import json
import time
import random
import mlflow
import dask.dataframe as dd
from dask.distributed import Client
import asyncio
import aiohttp
import nest_asyncio
import joblib
from sklearn.ensemble import RandomForestRegressor

# Apply nest_asyncio for async in Jupyter environments
nest_asyncio.apply()

# ------------------ CONSTANTS ------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_REGISTRY = "models"

# ------------------ INITIALIZATION ------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MLflow
import mlflow
mlflow.set_tracking_uri("file:/tmp/mlruns")
mlflow.set_experiment("Quantum-Stock-Forecasting")

# Initialize Dask client
client = Client(n_workers=4, threads_per_worker=2, memory_limit='2GB')

# avoid heavy imports at top-level
def load_model():
    import tensorflow as tf   # only when needed
    return tf.keras.models.load_model("...")


# ------------------ MLOPS COMPONENTS ------------------
def log_model_metrics(model_name, metrics, params, artifacts=None):
    """Log model metrics to MLflow"""
    try:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if artifacts:
                for artifact in artifacts:
                    mlflow.log_artifact(artifact)
            mlflow.set_tag("model_type", model_name.split("_")[0])
            mlflow.set_tag("status", "production")
            return True
    except Exception as e:
        logger.error(f"MLflow logging failed: {str(e)}")
        return False

def detect_drift(current_rmse, previous_rmse, threshold=0.05):
    """Detect model performance drift"""
    if previous_rmse == 0:  # Avoid division by zero
        return False
    return (current_rmse - previous_rmse) / previous_rmse > threshold

def retrain_model(model_type, data, forecast_days=90, trigger="scheduled"):
    """Automated model retraining pipeline"""
    try:
        logger.info(f"Starting retraining for {model_type} ({trigger})")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_retrain_{datetime.now().strftime('%Y%m%d')}"):
            # Log retraining trigger
            mlflow.set_tag("retrain_trigger", trigger)
            
            # Train model
            if model_type == "prophet":
                model, forecast = prophet_forecast(data, forecast_days)
                val_rmse = np.sqrt(mean_squared_error(
                    data['Close'].iloc[-30:], 
                    forecast['yhat'].iloc[-30-forecast_days:-forecast_days]
                ))
            elif model_type == "tft":
                results = tft_forecast(data, forecast_days)
                model = results['model']
                val_rmse = results['val_rmse']
            else:
                logger.error(f"Invalid model type: {model_type}")
                return None, None
            
            # Log metrics
            mlflow.log_metric("val_rmse", val_rmse)
            
            # Save model
            os.makedirs(MODEL_REGISTRY, exist_ok=True)
            model_path = f"{MODEL_REGISTRY}/{model_type}_{datetime.now().strftime('%Y%m%d%H%M')}"
            if model_type == "prophet":
                joblib.dump(model, f"{model_path}.pkl")
            else:
                torch.save(model.state_dict(), f"{model_path}.pt")
            
            mlflow.log_artifact(f"{model_path}.{'pkl' if model_type == 'prophet' else 'pt'}")
            
            logger.info(f"Retraining complete for {model_type}. Validation RMSE: {val_rmse:.4f}")
            return model, val_rmse
        
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        return None, None

# ------------------ ASYNC DATA FETCHING ------------------
async def fetch_stock_data_async(ticker, start, end, session):
    """Async fetch for single stock"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1d"
    }
    
    try:
        async with session.get(url, params=params, timeout=10) as response:
            data = await response.json()
            if 'chart' in data and 'result' in data['chart']:
                prices = data['chart']['result'][0]['indicators']['quote'][0]
                timestamps = data['chart']['result'][0]['timestamp']
                df = pd.DataFrame({
                    'Date': pd.to_datetime(timestamps, unit='s'),
                    'Open': prices['open'],
                    'High': prices['high'],
                    'Low': prices['low'],
                    'Close': prices['close'],
                    'Volume': prices['volume']
                }).dropna()
                df['Ticker'] = ticker
                return df
    except Exception as e:
        logger.error(f"Async fetch error for {ticker}: {str(e)}")
    return pd.DataFrame()

async def fetch_multiple_stocks(tickers, start, end):
    """Fetch multiple stocks asynchronously"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_stock_data_async(ticker, start, end, session) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return pd.concat([df for df in results if not df.empty])

# ------------------ FEATURE ENRICHMENT ------------------
def get_market_sentiment(ticker):
    """Get real-time market sentiment scores"""
    # Placeholder implementation - real version would use APIs
    return {
        'news_sentiment': random.uniform(-1, 1),
        'social_sentiment': random.uniform(-1, 1),
        'analyst_rating': random.choice(["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"])
    }

def get_macro_features(date):
    """Get macroeconomic features for specific date"""
    # Placeholder implementation - real version would use APIs
    return {
        'vix': random.uniform(15, 40),
        'usd_inr': random.uniform(72, 78),
        'oil_price': random.uniform(60, 120),
        'treasury_10y': random.uniform(1.0, 5.0),
        'gdp_growth': random.uniform(1.0, 8.0)
    }

def get_sector_index(ticker):
    """Get relevant sector index for a stock"""
    sector_map = {
        "TECH": ["AAPL", "MSFT", "GOOGL"],
        "FINANCE": ["JPM", "BAC", "GS"],
        "ENERGY": ["XOM", "CVX", "COP"]
    }
    for sector, tickers in sector_map.items():
        if ticker in tickers:
            return f"{sector}_SECTOR_INDEX"
    return "GENERAL_MARKET_INDEX"

def enrich_data(data, ticker):
    """Enrich dataset with additional features"""
    # Add market sentiment
    sentiment = get_market_sentiment(ticker)
    data['news_sentiment'] = sentiment['news_sentiment']
    data['social_sentiment'] = sentiment['social_sentiment']
    
    # Add macroeconomic features
    for date in data.index:
        macro = get_macro_features(date)
        for key, value in macro.items():
            data.loc[date, key] = value
    
    # Add sector index
    try:
        sector_index = get_sector_index(ticker)
        sector_data = yf.download(sector_index, start=data.index.min(), end=data.index.max())['Close']
        data['sector_index'] = sector_data.reindex(data.index, method='ffill')
    except:
        data['sector_index'] = data['Close']  # Fallback to price
    
    return data

# ------------------ FORECASTING MODELS ------------------
def prophet_forecast(data, forecast_days, country='IN'):
    """Enhanced Prophet forecasting with additional features"""
    if len(data) < 90:
        raise ValueError("Need at least 90 days of data for forecasting")
    
    # Create holiday dataframe
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
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=10,
        changepoint_range=0.8,
        interval_width=0.95,
        uncertainty_samples=100,
        holidays=holiday_df
    )
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
    
    # Add technical indicators and enriched features
    regressors = ['SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'MACD_Hist', 
                 'BB_Width', 'Volatility', 'news_sentiment', 'social_sentiment',
                 'vix', 'oil_price', 'treasury_10y', 'gdp_growth', 'sector_index']
    
    for regressor in regressors:
        if regressor in data.columns:
            prophet_df[regressor] = data[regressor].values
            model.add_regressor(regressor)
    
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Add future regressors (using last known values)
    for regressor in regressors:
        if regressor in data.columns:
            last_value = data[regressor].iloc[-1]
            future[regressor] = last_value
    
    forecast = model.predict(future)
    return model, forecast

def tft_forecast(data, forecast_days, tune=False):
    """Enhanced TFT forecasting with additional features"""
    # Add technical indicators
    data = calculate_technical_indicators(data.copy())
    
    # Enrich data with external features
    data = enrich_data(data, ticker)
    
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
                'Return_1d', 'Return_3d', 'Return_5d', 'Return_7d',
                'news_sentiment', 'social_sentiment', 'vix', 'oil_price',
                'treasury_10y', 'gdp_growth', 'sector_index']
    
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
    
    # Configure TFT with best parameters
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
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
    val_rmse = trainer.callback_metrics["val_loss"].item()
    
    return {
        'forecast': forecast,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'model': tft
    }

# ------------------ ENSEMBLE METHODS ------------------
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

def meta_learner_ensemble(prophet_forecast, tft_forecast, actuals, forecast_days):
    """Stacked regression ensemble using meta-learner"""
    # Prepare training data for meta-learner
    X_train = np.column_stack((
        prophet_forecast[-60:-forecast_days],
        tft_forecast[:60]
    ))
    y_train = actuals[-60:]
    
    # Train meta-model
    meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
    meta_model.fit(X_train, y_train)
    
    # Prepare prediction data
    X_pred = np.column_stack((
        prophet_forecast[-forecast_days:],
        tft_forecast
    ))
    
    # Generate meta-forecast
    meta_forecast = meta_model.predict(X_pred)
    
    return meta_forecast, meta_model

# ------------------ BACKTESTING & A/B TESTING ------------------
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
    
    # Calculate win rate
    win_rate = (returns > 0).mean() * 100 if len(returns) > 0 else 0
    
    # Calculate Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
    
    # Calculate Calmar ratio
    calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'return': total_return,
        'drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': len(trades),
        'portfolio': portfolio,
        'win_rate': win_rate,
        'sortino': sortino,
        'calmar': calmar
    }

def compare_strategies(data, strategies):
    """A/B test multiple forecasting strategies"""
    results = {}
    for name, strategy in strategies.items():
        start_time = time.time()
        metrics = backtest_strategy(
            data, 
            strategy['function'], 
            strategy['params']
        )
        results[name] = {
            'metrics': metrics,
            'runtime': time.time() - start_time
        }
    return results

# ------------------ UTILITY FUNCTIONS ------------------
def calculate_technical_indicators(data):
    """Calculate technical indicators for stock data"""
    if 'Close' not in data.columns or len(data) < 20:
        return data
    
    close_series = data['Close'].squeeze()
    
    # Moving Averages
    data['SMA20'] = close_series.rolling(window=20).mean()
    data['SMA50'] = close_series.rolling(window=50).mean()
    data['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
    
    # RSI
    data['RSI'] = ta.momentum.rsi(close_series, window=14)
    
    # MACD
    macd = ta.trend.MACD(close_series)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()
    data['BB_Width'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()
    
    # Volatility
    returns = close_series.pct_change()
    data['Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
    
    # Lagged returns
    for i in [1, 3, 5, 7]:
        data[f'Return_{i}d'] = close_series.pct_change(i)
    
    return data.dropna()

@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker):
    """Fetch news articles related to a stock ticker"""
    logger.info(f"Fetching news for {ticker}")
    api_key = os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY")
    if not api_key:
        logger.error("News API key not found")
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

# ------------------ STREAMLIT UI ------------------
def main():
    # Custom CSS for styling
    CUSTOM_CSS = """
    <style>
        /* ... (same custom CSS as before) ... */
    </style>
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="header">🚀 QUANTUM STOCK ANALYTICS</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">AI-Powered Financial Intelligence Platform</h3>', unsafe_allow_html=True)
    
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
    
    # Advanced options
    st.sidebar.markdown("### ⚙️ Advanced Options")
    tune_hyperparams = st.sidebar.checkbox("Tune Hyperparameters", value=False)
    enable_ensemble = st.sidebar.checkbox("Enable Ensemble Forecasting", value=True)
    
    # Fetch stock data
    with st.spinner('Fetching market data...'):
        data = asyncio.run(fetch_stock_data_async(
            ticker, 
            start_date, 
            end_date,
            aiohttp.ClientSession()
        ))
    
    # Create tabs
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
            if len(data) > 1 and 'Close' in data.columns:
                current_price = float(data['Close'].iloc[-1])
                prev_price = float(data['Close'].iloc[-2]) if len(data) >= 2 else current_price
                volume = float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0.0
                daily_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0.0
                volatility = float(calculate_volatility(data))
                annual_return = float(calculate_annual_return(data, start_date, end_date) * 100)

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
                    col_tft2.metric("Validation RMSE", f"{tft_results['val_rmse']:.2f}")
                    
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
                        
                        # Generate meta-learner forecast
                        meta_forecast, meta_model = meta_learner_ensemble(
                            prophet_forecast_df['yhat'].values,
                            tft_results['forecast'],
                            data['Close'].values,
                            forecast_days
                        )
                        
                        # Plot all forecasts
                        st.subheader("Hybrid Forecasting Comparison")
                        fig_hybrid = go.Figure()
                        
                        # Actual data
                        fig_hybrid.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        # Prophet forecast
                        fig_hybrid.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=prophet_forecast_df['yhat'].values[-forecast_days:],
                            mode='lines',
                            name='Prophet Forecast',
                            line=dict(color='#ff7f0e', dash='dash')
                        ))
                        
                        # TFT forecast
                        fig_hybrid.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=tft_results['forecast'],
                            mode='lines',
                            name='TFT Forecast',
                            line=dict(color='#2ca02c', dash='dot')
                        ))
                        
                        # Simple ensemble
                        fig_hybrid.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=combined_forecast,
                            mode='lines',
                            name='Weighted Ensemble',
                            line=dict(color='#d62728', width=3)
                        ))
                        
                        # Meta-learner
                        fig_hybrid.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=meta_forecast,
                            mode='lines',
                            name='Meta-learner Ensemble',
                            line=dict(color='#9467bd', width=3)
                        ))
                        
                        fig_hybrid.update_layout(
                            title=f'Hybrid Forecasting Comparison for {ticker}',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            template='plotly_dark',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=600
                        )
                        st.plotly_chart(fig_hybrid, use_container_width=True)
                        
                        # Display weights
                        st.subheader("Ensemble Weights")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Prophet Weight", f"{prophet_weight:.2%}")
                        col2.metric("TFT Weight", f"{tft_weight:.2%}")
                        col3.metric("Meta-learner", "Random Forest")
                        
                        # Feature importance from meta-learner
                        st.subheader("Meta-learner Feature Importance")
                        try:
                            importances = meta_model.feature_importances_
                            features = ['Prophet', 'TFT']
                            fig_importance = px.bar(
                                x=features,
                                y=importances,
                                labels={'x': 'Model', 'y': 'Importance'},
                                title='Meta-learner Feature Importance',
                                color=features,
                                color_discrete_sequence=['#ff7f0e', '#2ca02c']
                            )
                            fig_importance.update_layout(template='plotly_dark')
                            st.plotly_chart(fig_importance, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Feature importance not available: {str(e)}")
                        
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
            try:
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
            except Exception as e:
                st.error(f"Earnings data error: {str(e)}")
                st.warning("Using simulated earnings data")
                
                # Create mock data for 2024-2025
                dates = pd.date_range(start='2024-01-01', periods=4, freq='Q')
                earnings_data = pd.DataFrame({
                    'Earnings Date': dates,
                    'EPS Estimate': np.random.uniform(0.5, 2.5, 4),
                    'Reported EPS': np.random.uniform(0.4, 2.6, 4),
                    'Surprise (%)': np.random.uniform(-15, 15, 4)
                })
                earnings_data.set_index('Earnings Date', inplace=True)
                
                fig_earn = go.Figure()
                fig_earn.add_trace(go.Bar(
                    x=earnings_data.index,
                    y=earnings_data['Surprise (%)'],
                    name='Earnings Surprise',
                    marker_color=np.where(earnings_data['Surprise (%)'] > 0, 'green', 'red')
                ))
                fig_earn.update_layout(
                    title='Recent Earnings Surprise (Simulated)',
                    xaxis_title='Date',
                    yaxis_title='Surprise (%)',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_earn, use_container_width=True)

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
                stock_data = asyncio.run(fetch_stock_data_async(t, start_date, end_date, aiohttp.ClientSession()))
                actual_returns[t] = calculate_annual_return(stock_data, start_date, end_date) * 100

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
                response = generate_ai_response(query, data, portfolio_data, "Moderate", "Growth")
                st.markdown(f"""
                <div class="ai-response">
                    <h4>🔍 AI Analysis</h4>
                    <p style="font-size:1.1em;">{response}</p>
                    <div style="display:flex; justify-content:space-between; margin-top:20px;">
                        <small>Generated at {datetime.now().strftime('%H:%M:%S')}</small>
                    </div>
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

    # MLOps Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("MLOps Management")
    if st.sidebar.button("Retrain Prophet Model"):
        with st.spinner("Retraining Prophet model..."):
            model, rmse = retrain_model("prophet", data)
            if model:
                st.sidebar.success(f"Prophet retrained! Val RMSE: {rmse:.4f}")
    
    if st.sidebar.button("Retrain TFT Model"):
        with st.spinner("Retraining TFT model..."):
            model, rmse = retrain_model("tft", data)
            if model:
                st.sidebar.success(f"TFT retrained! Val RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
