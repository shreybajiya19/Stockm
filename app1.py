import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from neuralprophet import NeuralProphet
import yfinance as yf
from datetime import datetime, timedelta

def main():
    st.title('Stock Price Prediction')

    # Back button with improved styling and same tab behavior
    back_button = st.markdown("""
        <style>
            .back-button {
                background-color: transparent;
                color: white;
                border: 2px solid white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 12px;
            }
            .back-button:hover {
                background-color: white;
                color: black;
            }
        </style>
        <a href="https://techandtheories.in" class="back-button">Back</a>
    """, unsafe_allow_html=True)

    # User input for stock symbol
    stock_symbol = st.text_input('Enter the stock symbol (e.g., AAPL):', '').strip().upper()

    if not stock_symbol:
        st.error('Please enter a valid stock symbol.')
        return

    # Define the start date and end date
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years of data
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Download stock data from Yahoo Finance
    @st.cache  # Cache data to avoid fetching repeatedly during the session
    def load_data(symbol, start, end):
        stock_data = yf.download(symbol, start=start, end=end, progress=False)
        return stock_data

    with st.spinner(f'Loading data for {stock_symbol}...'):
        stock_data = load_data(stock_symbol, start_date, end_date)

    if stock_data.empty:
        st.error(f'Failed to download data for {stock_symbol}. Please check the stock symbol and try again.')
        return

    st.write(f"Stock Data for {stock_symbol}")
    st.write(stock_data)

    # Prepare data for NeuralProphet
    stocks = stock_data[['Close']].reset_index()
    stocks.columns = ['ds', 'y']  # NeuralProphet expects columns named 'ds' (date) and 'y' (target)

    # Initialize NeuralProphet model with optimized parameters
    model = NeuralProphet(
        growth='linear',  # linear or logistic
        changepoints=None,  # list of dates at which to include potential changepoints
        n_changepoints=5,  # number of potential changepoints to include
        yearly_seasonality='auto',  # can be 'auto', True, False, or a number of Fourier components to generate
        weekly_seasonality='auto',  # same as above
        daily_seasonality='auto',  # same as above
        seasonality_mode='additive',  # 'additive' or 'multiplicative'
        seasonality_reg=0.0,  # strength of the seasonality prior
        n_lags=0,  # number of lagged variables to include as additional features
        num_hidden_layers=0,  # number of hidden layers to include in the AR-Net component
        d_hidden=10,  # dimensionality of the hidden layers
        ar_sparsity=None,  # use Auto AR Sparsity
        learning_rate=1.0,  # learning rate parameter
        epochs=40,  # number of epochs to train the model
    )

    # Fit the model
    with st.spinner('Training model...'):
        model.fit(stocks, freq='D')  # Assuming daily data

    # Make future predictions
    future = model.make_future_dataframe(stocks, periods=300)
    forecast = model.predict(future)
    actual_prediction = model.predict(stocks)

    # Plotting with Plotly
    st.subheader('Stock Price Prediction Results')

    # Create traces for actual data, predictions on actual data, and future forecasts
    actual_trace = go.Scatter(
        x=stocks['ds'], y=stocks['y'], mode='lines', name='Actual', line=dict(color='green'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
    )

    prediction_trace = go.Scatter(
        x=actual_prediction['ds'], y=actual_prediction['yhat1'], mode='lines', name='Predicted on Actual', line=dict(color='red'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
    )

    forecast_trace = go.Scatter(
        x=forecast['ds'], y=forecast['yhat1'], mode='lines', name='Future Forecast', line=dict(color='blue'),
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
    fig = go.Figure(data=[actual_trace, prediction_trace, forecast_trace], layout=layout)

    # Display Plotly chart
    st.plotly_chart(fig)

    # Add instruction text just below the legend using Markdown
    st.markdown('<div style="text-align: center; margin-top: -20px;">Click on the legend to plot the graph.</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
