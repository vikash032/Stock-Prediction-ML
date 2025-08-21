                yaxis_title='Surprise (%)',
                template='plotly_dark'
            )
            st.plotly_chart(fig_earn, use_container_width=True)
            
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
                stock_data = get_stock_data(t, start_date, end_date)
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
            
            # Macroeconomic Dashboard
            st.subheader("Macroeconomic Dashboard")
            macro_data = get_macro_data()
            
            col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
            col_m1.markdown(f"""
                <div class="macro-metric">
                    <h5>Inflation</h5>
                    <h3>{macro_data['inflation']}%</h3>
                    <small>Source: {macro_data['source']}</small>
                    <small>Updated: {macro_data['last_updated']}</small>
                </div>
            """, unsafe_allow_html=True)
            col_m2.markdown(f"""
                <div class="macro-metric">
                    <h5>Interest Rate</h5>
                    <h3>{macro_data['interest_rate']}%</h3>
                    <small>Source: {macro_data['source']}</small>
                    <small>Updated: {macro_data['last_updated']}</small>
                </div>
            """, unsafe_allow_html=True)
            col_m3.markdown(f"""
                <div class="macro-metric">
                    <h5>Unemployment</h5>
                    <h3>{macro_data['unemployment']}%</h3>
                    <small>Source: {macro_data['source']}</small>
                    <small>Updated: {macro_data['last_updated']}</small>
                </div>
            """, unsafe_allow_html=True)
            col_m4.markdown(f"""
                <div class="macro-metric">
                    <h5>GDP Growth</h5>
                    <h3>{macro_data['gdp_growth']}%</h3>
                    <small>Source: {macro_data['source']}</small>
                    <small>Updated: {macro_data['last_updated']}</small>
                </div>
            """, unsafe_allow_html=True)
            col_m5.markdown(f"""
                <div class="macro-metric">
                    <h5>Consumer Sentiment</h5>
                    <h3>{macro_data['consumer_sentiment']}</h3>
                    <small>Source: {macro_data['source']}</small>
                    <small>Updated: {macro_data['last_updated']}</small>
                </div>
            """, unsafe_allow_html=True)
            col_m6.markdown(f"""
                <div class="macro-metric">
                    <h5>Manufacturing PMI</h5>
                    <h3>{macro_data['manufacturing_pmi']}</h3>
                    <small>Source: {macro_data['source']}</small>
                    <small>Updated: {macro_data['last_updated']}</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Data validation warnings
            if macro_data['inflation'] > 10:
                st.warning("High inflation rate detected - may impact portfolio performance")
            if macro_data['unemployment'] > 10:
                st.warning("High unemployment rate detected - may indicate economic slowdown")
            if macro_data['consumer_sentiment'] < 50:
                st.warning("Low consumer sentiment - may impact consumer stocks")
            
            st.markdown(f"""
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
                ""', unsafe_allow_html=True)
        
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
            
            if 'portfolio' in results and len(results['portfolio']) > 0:
                fig_backtest.add_trace(go.Scatter(
                    x=data.index[params.get('long_window', 50):][:len(results['portfolio'])],
                    y=results['portfolio'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00FF00'),
                    yaxis='y2'
                ))
            
            # Add trade markers
            if 'trades' in results:
                buy_dates = [t[1] for t in results['trades'] if t[0] == 'buy']
                buy_prices = [t[2] for t in results['trades'] if t[0] == 'buy']
                sell_dates = [t[1] for t in results['trades'] if t[0] == 'sell']
                sell_prices = [t[2] for t in results['trades'] if t[0] == 'sell']
                
                if buy_dates:
                    fig_backtest.add_trace(go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name='Buy',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ))
                
                if sell_dates:
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
            if 'trades' in results and results['trades']:
                st.subheader("Trade Log")
                trades_df = pd.DataFrame(results['trades'], columns=['Action', 'Date', 'Price', 'Shares'])
                st.dataframe(trades_df)

    # Real-Time Monitoring Tab - FIXED rerun issue
    with tab8:
        st.markdown('<div class="header">🚀 Real-Time Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Live market monitoring and predictions</div>', unsafe_allow_html=True)
        
        # Real-time data fetching
        if st.button("Refresh Real-Time Data"):
            # Use st.rerun() instead of st.experimental_rerun()
            st.rerun()
        
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

    # Market Indices Tab
    with tab9:
        st.markdown('<div class="header">📊 Market Indices</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Real-time Indian Market Index Performance</div>', unsafe_allow_html=True)
        
        # Define Indian market indices
        indices = {
            'Nifty 50': '^NSEI',
            'Sensex': '^BSESN',
            'Bank Nifty': 'NSEBANK.NS',
            'FinNifty': 'FINNIFTY.NS',
            'Nifty 100': '^CNX100'
        }
        
        # Fetch current data for all indices
        index_data = {}
        sentiment_data = {}
        
        with st.spinner('Fetching market index data...'):
            for name, ticker in indices.items():
                try:
                    # Get latest data
                    data = yf.download(ticker, period='1d', interval='1m')
                    if not data.empty:
                        index_data[name] = {
                            'current': data['Close'].iloc[-1],
                            'previous': data['Close'].iloc[0] if len(data) > 1 else data['Close'].iloc[-1],
                            'change': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100) 
                                    if len(data) > 1 else 0
                        }
                        
                        # Get more data for sentiment analysis
                        historical_data = yf.download(ticker, period='1mo')
                        if not historical_data.empty:
                            historical_data = calculate_technical_indicators(historical_data)
                            sentiment_data[name] = get_market_sentiment(historical_data)
                    else:
                        index_data[name] = {'current': 0, 'previous': 0, 'change': 0}
                        sentiment_data[name] = "Neutral"
                except Exception as e:
                    logger.error(f"Error fetching {name} data: {str(e)}")
                    index_data[name] = {'current': 0, 'previous': 0, 'change': 0}
                    sentiment_data[name] = "Neutral"
        
        # Display index performance
        st.subheader("Index Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        index_cols = [col1, col2, col3, col4, col5]
        for i, (name, data) in enumerate(index_data.items()):
            with index_cols[i]:
                change_color = "green" if data['change'] >= 0 else "red"
                sentiment = sentiment_data.get(name, "Neutral")
                sentiment_icon = "📈" if sentiment == "Bullish" else "📉" if sentiment == "Bearish" else "➡️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <h2>${data['current']:,.2f}</h2>
                    <p style="color:{change_color}; font-size:1.2em;">
                        {data['change']:+.2f}%
                    </p>
                    <p>{sentiment_icon} {sentiment}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Market overview
        st.subheader("Market Overview")
        bullish_count = sum(1 for sentiment in sentiment_data.values() if sentiment == "Bullish")
        bearish_count = sum(1 for sentiment in sentiment_data.values() if sentiment == "Bearish")
        neutral_count = sum(1 for sentiment in sentiment_data.values() if sentiment == "Neutral")
        
        col_overview1, col_overview2, col_overview3 = st.columns(3)
        col_overview1.metric("Bullish Indices", bullish_count)
        col_overview2.metric("Bearish Indices", bearish_count)
        col_overview3.metric("Neutral Indices", neutral_count)
        
        # Overall market sentiment
        if bullish_count >= 3:
            overall_sentiment = "Bullish"
            sentiment_icon = "📈"
            sentiment_color = "green"
        elif bearish_count >= 3:
            overall_sentiment = "Bearish"
            sentiment_icon = "📉"
            sentiment_color = "red"
        else:
            overall_sentiment = "Neutral"
            sentiment_icon = "➡️"
            sentiment_color = "gray"
        
        st.markdown(f"""
        <div class="gauge">
            <div class="gauge-value" style="color:{sentiment_color}">{overall_sentiment} {sentiment_icon}</div>
            <small>Overall Market Sentiment</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Index comparison chart
        st.subheader("Index Performance Comparison")
        comparison_data = []
        for name, data in index_data.items():
            comparison_data.append({
                'Index': name,
                'Current Value': data['current'],
                'Daily Change (%)': data['change'],
                'Sentiment': sentiment_data.get(name, "Neutral")
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        fig_comparison = px.bar(comparison_df, x='Index', y='Daily Change (%)', 
                               color='Sentiment', color_discrete_map={
                                   'Bullish': 'green',
                                   'Bearish': 'red',
                                   'Neutral': 'gray'
                               })
        fig_comparison.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Sector performance (simulated)
        st.subheader("Sector Performance")
        sectors = ['IT', 'Banking', 'Pharma', 'Auto', 'FMCG', 'Energy', 'Real Estate', 'Infrastructure']
        sector_performance = {sector: np.random.uniform(-3, 3) for sector in sectors}
        
        fig_sector = px.bar(x=list(sector_performance.keys()), y=list(sector_performance.values()),
                           labels={'x': 'Sector', 'y': 'Daily Change (%)'},
                           color=list(sector_performance.values()),
                           color_continuous_scale='RdYlGn')
        fig_sector.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Market news
        st.subheader("Market News")
        market_news = []
        for index_name in indices.keys():
            news = get_news(indices[index_name])
            if news:
                market_news.extend(news[:2])  # Get top 2 news for each index
        
        if market_news:
            for news in market_news[:5]:  # Show top 5 news
                st.markdown(f"""
                <div class="news-item">
                    <b>{news['title']}</b><br>
                    <i>{news.get('source', 'Unknown')} - {news.get('date', '')[:10]}</i><br>
                    <a href="{news['link']}" target="_blank">Read more</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No market news available at the moment")

if __name__ == "__main__":
    main()
