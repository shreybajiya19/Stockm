import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Normalization

def main():
    st.title('Stock Price Prediction with LSTM')

    # Add a back button at the top of the page
    if st.button('Back'):
        st.experimental_set_query_params(page='index.html')

    # User input for stock symbol
    stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL):', 'AAPL').upper()  # Default to AAPL

    # Define the start date (from March 2020)
    start_date = '2020-03-01'

    # Get today's date for the end date
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Download stock data from Yahoo Finance
    @st.cache_data  # Cache data to avoid fetching repeatedly during the session
    def load_data(symbol, start, end):
        stock_data = yf.download(symbol, start=start, end=end)
        return stock_data

    with st.spinner('Loading data...'):
        stock_data = load_data(stock_symbol, start_date, end_date)

    if stock_data.empty:
        st.error('Failed to download data. Please check the stock symbol and try again.')
        return

    st.write("Stock Data")
    st.write(stock_data)

    # Prepare data for LSTM
    data = stock_data['Close'].values.reshape(-1, 1)
    
    # Use TensorFlow's Normalization layer
    normalizer = Normalization()
    normalizer.adapt(data)
    scaled_data = normalizer(data)

    # Ensure enough data for training and testing
    train_size = int(len(scaled_data) * 0.8)  # 80% for training
    test_size = len(scaled_data) - train_size  # Remaining for testing

    if len(scaled_data) < 100 or train_size < 60 or test_size < 20:
        st.error("Not enough data available for prediction. Please choose another stock symbol or timeframe.")
        return

    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Check shapes before reshaping
    st.write(f"X_train shape before reshaping: {X_train.shape}")
    st.write(f"X_test shape before reshaping: {X_test.shape}")

    # Reshape input to be [samples, time steps, features]
    try:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    except IndexError as e:
        st.error(f"Error reshaping data: {e}")
        return

    # Check shapes after reshaping
    st.write(f"X_train shape after reshaping: {X_train.shape}")
    st.write(f"X_test shape after reshaping: {X_test.shape}")

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    with st.spinner('Training model...'):
        model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions using the normalizer's inverse_transform
    train_predict = normalizer.mean + normalizer.variance ** 0.5 * train_predict
    test_predict = normalizer.mean + normalizer.variance ** 0.5 * test_predict
    actual_data = normalizer.mean + normalizer.variance ** 0.5 * scaled_data

    # Plotting with Plotly
    st.subheader('Stock Price Prediction Results')

    # Create traces for actual data and predictions
    actual_trace = go.Scatter(
        x=stock_data.index, y=actual_data.flatten(), mode='lines', name='Actual', line=dict(color='green'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
    )

    train_trace = go.Scatter(
        x=stock_data.index[:len(train_predict)], y=train_predict.flatten(), mode='lines', name='Train Predict', line=dict(color='red'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
    )

    test_trace = go.Scatter(
        x=stock_data.index[len(train_predict)+(time_step*2)+1:len(scaled_data)-1], y=test_predict.flatten(), mode='lines', name='Test Predict', line=dict(color='blue'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
    )

    # Customize layout
    layout = go.Layout(
        title=f'Stock Price Prediction for {stock_symbol}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price (in $)'),
        hovermode='x unified',  # Shows values for all traces at the hovered x-coordinate
        hoverlabel=dict(bgcolor="rgba(240, 240, 240, 0.8)", font_size=12, font_family="Arial", font_color="black"),
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True
    )

    # Create figure and add traces
    fig = go.Figure(data=[actual_trace, train_trace, test_trace], layout=layout)

    # Display Plotly chart
    st.plotly_chart(fig)

    # Add instruction text just below the legend
    st.write('<div style="text-align: center; margin-top: -20px;">Click on the legend to plot the graph.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
