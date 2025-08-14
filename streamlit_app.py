import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, date
from pandas.tseries.offsets import BDay

# Sidebar controls for user input
instrument = st.sidebar.text_input("Instrument (Ticker)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
show_ma = st.sidebar.checkbox("Show Moving Average", value=True)
ma_window = st.sidebar.slider("SMA Window", 3, 30, 10)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_pattern = st.sidebar.checkbox("Detect Peaks/Troughs", value=True)
show_ml = st.sidebar.checkbox("Predict ML Trend", value=False)
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

@st.cache_data
def fetch_historical_data_yf(ticker, start, end):
    # Download historical data from Yahoo Finance
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return None

    # Flatten MultiIndex columns if present (e.g., ('Close', 'AAPL'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]  # keep first level, lowercase
    else:
        df.columns = [col.lower() for col in df.columns]

    df = df.reset_index()  # Move Date from index to column

    # Rename 'Date' column from reset_index to 'date', if present
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    elif 'date' not in df.columns:
        # If no 'Date' column and 'date' missing, rename first column to 'date'
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing essential data
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    df = df.sort_values('date')
    return df

# Fetch historical data for the selected instrument and date range
df = fetch_historical_data_yf(instrument, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if df is not None and not df.empty:
    # Ensure date column is datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # Convert price and volume columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to numeric: {e}")

    # Drop rows with missing essential data
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Prepare data for plotting
    dates = df['date']
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']

    # Create candlestick chart data
    plotly_data = [
        go.Candlestick(
            x=dates, open=open_prices, high=high_prices,
            low=low_prices, close=close_prices, name="Candles"
        )
    ]

    # Add moving average line if selected
    if show_ma:
        df['ma'] = df['close'].rolling(window=ma_window).mean()
        plotly_data.append(go.Scatter(
            x=dates, y=df['ma'], mode='lines',
            name=f"SMA{ma_window}", line=dict(color='blue', width=1)
        ))

    # Add volume bars if selected
    if show_volume and 'volume' in df.columns:
        plotly_data.append(go.Bar(
            x=dates, y=df['volume'], name='Volume', marker_color='orange', opacity=0.3, yaxis='y2'
        ))

    # Detect and plot peaks and troughs if selected
    if show_pattern:
        close_np = df['close'].values
        peaks, troughs = [], []
        for i in range(1, len(close_np) - 1):
            if close_np[i] > close_np[i - 1] and close_np[i] > close_np[i + 1]:
                peaks.append(i)
            if close_np[i] < close_np[i - 1] and close_np[i] < close_np[i + 1]:
                troughs.append(i)
        if peaks:
            plotly_data.append(go.Scatter(
                x=dates.iloc[peaks], y=df['close'].iloc[peaks],
                mode='markers', name='Peaks', marker=dict(symbol='triangle-up', color='red', size=10)
            ))
        if troughs:
            plotly_data.append(go.Scatter(
                x=dates.iloc[troughs], y=df['close'].iloc[troughs],
                mode='markers', name='Troughs', marker=dict(symbol='triangle-down', color='green', size=10)
            ))

    # Add ML-based linear regression forecast if selected
    if show_ml and len(df) > 30:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values
        model = LinearRegression().fit(X, y)
        fut_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
        last_date = df['date'].iloc[-1]
        start_forecast_date = last_date + BDay(1)
        fut_dates = pd.date_range(start=start_forecast_date, periods=forecast_days, freq='B')
        future_preds = model.predict(fut_X)
        plotly_data.append(go.Scatter(
            x=fut_dates, y=future_preds, mode='lines',
            name="ML Forecast", line=dict(color='magenta', dash='dash')
        ))

    # Create and display the Plotly figure
    fig = go.Figure(
        data=plotly_data,
        layout=go.Layout(
            title=f'Candlestick Chart for {instrument}',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.12),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template="plotly_white"
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    # Show last 20 rows of the dataframe
    st.dataframe(df.tail(20))

else:
    # Show error if data could not be fetched
    st.error("Failed to fetch historical data for this ticker and date range.")