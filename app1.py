import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from neuralprophet import NeuralProphet
import yfinance as yf
from datetime import datetime, timedelta

def main():
    st.title('Stock Price and Financial Data Analysis')

    # Back button
    st.markdown(
        '<a href="https://techandtheories.in" target="_blank"><button style="background-color:#4CAF50;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;">Back</button></a>', 
        unsafe_allow_html=True
    )

    # User input for stock symbol
    stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL):').upper()

    # User input for number of days to predict (initially blank)
    predict_days = st.number_input('Enter number of days to predict:', min_value=1, value=1, step=1, format='%d')

    # Predict button
    if st.button('Predict'):
        if not stock_symbol:
            st.error('Please enter a stock symbol.')
            return

        # Define the start date
        start_date = '2000-01-01'

        # Get today's date for the end date
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Download stock data from Yahoo Finance
        @st.cache
        def load_data(symbol, start, end):
            stock_data = yf.download(symbol, start=start, end=end, progress=False)
            return stock_data

        with st.spinner('Loading stock data...'):
            stock_data = load_data(stock_symbol, start_date, end_date)

        if stock_data.empty:
            st.error('Failed to download data. Please check the stock symbol and try again.')
            return

        st.write("Stock Data")
        st.write(stock_data)

        # Download additional financial data from Yahoo Finance
        @st.cache
        def load_financial_data(symbol):
            ticker = yf.Ticker(symbol)
            financials = {
                'Balance Sheet': ticker.balance_sheet,
                'Income Statement': ticker.financials,
                'Cash Flow': ticker.cashflow,
                'Ratios': ticker.financials.loc[['Gross Profit', 'Operating Income', 'Net Income']],
            }
            return financials

        with st.spinner('Loading financial data...'):
            financial_data = load_financial_data(stock_symbol)

        # Display financial data
        for title, data in financial_data.items():
            st.subheader(title)
            st.write(data)

        # Prepare data for NeuralProphet
        stocks = stock_data[['Close']].reset_index()
        stocks.columns = ['ds', 'y']

        # Initialize NeuralProphet model
        model = NeuralProphet()

        # Fit the model
        with st.spinner('Training model...'):
            model.fit(stocks)

        # Make future predictions
        future = model.make_future_dataframe(stocks, periods=predict_days)
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
            hovermode='x unified',
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





# import streamlit as st
# import pandas as pd
# import plotly.graph_objs as go
# from neuralprophet import NeuralProphet
# import yfinance as yf
# from datetime import datetime

# def main():
#     st.title('Stock Price Prediction')

#     # Back button
#     back_button = st.markdown('<a href="https://techandtheories.in" target="_blank"><button style="background-color:#4CAF50;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;">Back</button></a>', unsafe_allow_html=True)

#     # User input for stock symbol
#     stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL):', 'AAPL').upper()  # Default to AAPL

#     # Define the start date
#     start_date = '2000-01-01'

#     # Get today's date for the end date
#     end_date = datetime.today().strftime('%Y-%m-%d')

#     # Download stock data from Yahoo Finance
#     @st.cache  # Cache data to avoid fetching repeatedly during the session
#     def load_data(symbol, start, end):
#         stock_data = yf.download(symbol, start=start, end=end, progress=False)  # Disable progress bar
#         return stock_data

#     with st.spinner('Loading data...'):
#         stock_data = load_data(stock_symbol, start_date, end_date)

#     if stock_data.empty:
#         st.error('Failed to download data. Please check the stock symbol and try again.')
#         return

#     st.write("Stock Data")
#     st.write(stock_data)

#     # Prepare data for NeuralProphet
#     stocks = stock_data[['Close']].reset_index()
#     stocks.columns = ['ds', 'y']  # NeuralProphet expects columns named 'ds' (date) and 'y' (target)

#     # Initialize NeuralProphet model
#     model = NeuralProphet()

#     # Fit the model
#     with st.spinner('Training model...'):
#         model.fit(stocks)

#     # Make future predictions
#     future = model.make_future_dataframe(stocks, periods=1000)  # Extend the dataframe for future periods
#     forecast = model.predict(future)  # Make predictions for the future
#     actual_prediction = model.predict(stocks)  # Predictions on the actual data

#     # Plotting with Plotly
#     st.subheader('Stock Price Prediction Results')

#     # Create traces for actual data, predictions on actual data, and future forecasts
#     actual_trace = go.Scatter(
#         x=stocks['ds'], y=stocks['y'], mode='lines', name='Actual', line=dict(color='green'),
#         hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
#     )

#     prediction_trace = go.Scatter(
#         x=actual_prediction['ds'], y=actual_prediction['yhat1'], mode='lines', name='Predicted on Actual', line=dict(color='red'),
#         hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
#     )

#     forecast_trace = go.Scatter(
#         x=forecast['ds'], y=forecast['yhat1'], mode='lines', name='Future Forecast', line=dict(color='blue'),
#         hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}'
#     )

#     # Customize layout
#     layout = go.Layout(
#         title=f'Stock Price Prediction for {stock_symbol}',
#         xaxis=dict(title='Date'),
#         yaxis=dict(title='Stock Price (in $)'),
#         hovermode='x unified',  # Shows values for all traces at the hovered x-coordinate
#         hoverlabel=dict(bgcolor="rgba(240, 240, 240, 0.8)", font_size=12, font_family="Arial", font_color="black"),
#         legend=dict(yanchor="top", y=1.02, xanchor="left", x=0),
#         margin=dict(l=40, r=40, t=40, b=40),
#         showlegend=True
#     )

#     # Create figure and add traces
#     fig = go.Figure(data=[actual_trace, prediction_trace, forecast_trace], layout=layout)

#     # Display Plotly chart
#     st.plotly_chart(fig)

#     # Add instruction text just below the legend using Markdown
#     st.markdown('<div style="text-align: center; margin-top: -20px;">Click on the legend to plot the graph.</div>', unsafe_allow_html=True)

# if __name__ == '__main__':
#     main()
