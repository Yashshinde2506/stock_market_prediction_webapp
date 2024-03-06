import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = stock_data[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df

# Function to train the Prophet model
def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

# Function to make the forecast
def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Function to calculate performance metrics
def calculate_performance_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# Streamlit app
def main():
    # Use st.session_state instead of a custom SessionState class
    if 'login_successful' not in st.session_state:
        st.session_state.login_successful = False

    st.sidebar.header('Navigation')
    page = st.sidebar.radio("Pick one", ["Home", "User Login"])

    if page == "Home":
        st.title('Stock Prediction')
        image_path1 = './img.jpg'

        # Create three columns
        col1, col2, col3 = st.columns(3)

        # Display images in each column
        col1.image(image_path1, width=300)

        st.markdown("""
        ## Welcome to the Stock Forecasting App!
        This app leverages the Prophet forecasting model, developed by Meta (formerly Facebook), to predict future stock prices. 
        Prophet is designed for analyzing time series data with strong seasonal effects and several seasons of historical data.
        
        ### How to Use:
        1. Enter the **ticker symbol** of the stock you're interested in (e.g., 'AAPL' for Apple Inc.).
        2. Choose the **start and end dates** for historical data analysis. It is recommended to include as much historical data as possible to enhance the accuracy of the forecast.
        3. Select the **forecast horizon** from the dropdown to predict 1, 2, 3, or 5 years into the future.
        4. Click the **"Forecast Stock Prices"** button to generate the forecast.
        The more historical data provided, the more accurately Prophet can capture and forecast seasonal patterns in the data.
        
        Scroll down to view the forecast results and performance metrics of the model.
        """)

    elif page == "User Login":
        

        if not st.session_state.login_successful:
            st.title('Login')
            col1, col2 = st.columns(2)
            
            user1 = col1.text_input('Username')
            pass1 = col2.text_input('Password', type='password')
            submit = st.button('Submit')

            if submit:
                if user1 == 'Yash' and pass1 == 'Pass123':
                    st.session_state.login_successful = True

        if st.session_state.login_successful:
            # Hide login input boxes and password input box after successful login
            st.empty()

            st.header('User Input Parameters')
            col1, col2 = st.columns(2)
            ticker_symbol = col1.text_input('Enter Ticker Symbol', 'RACE')
            start_date = col2.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
            end_date = col2.date_input('End Date', value=pd.to_datetime('today'))
            forecast_horizon = col1.selectbox('Forecast Horizon', options=['1 year', '2 years', '3 years', '5 years'], format_func=lambda x: x.capitalize())

            # Introduction
            # Set up the layout
            # Convert the selected horizon to days
            horizon_mapping = {'1 year': 365, '2 years': 730, '3 years': 1095, '5 years': 1825}
            forecast_days = horizon_mapping[forecast_horizon]

            if st.button('Forecast Stock Prices'):
                with st.spinner('Fetching data...'):
                    df = fetch_stock_data(ticker_symbol, start_date, end_date)

                with st.spinner('Training model...'):
                    model = train_prophet_model(df)
                    forecast = make_forecast(model, forecast_days)

                # Show input parameters
                st.subheader('User Input Parameters')
                col3, col4 = st.columns(2)
                col3.write(f'Ticker Symbol: {ticker_symbol}')
                col3.write(f'Start Date: {start_date}')
                col4.write(f'End Date: {end_date}')
                col4.write(f'Forecast Horizon: {forecast_horizon}')

                # Show forecast data
                st.subheader('Forecast Data')
                st.write('The table below shows the forecasted stock prices along with the lower and upper bounds of the predictions.')
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Predicted', 'yhat_lower': 'Lower Limit', 'yhat_upper': 'Upper Limit'})
                st.write(forecast_table.head())

                st.subheader('Forecast Plot')
                st.write('The plot below visualizes the predicted stock prices with their confidence intervals.')
                fig1 = plot_plotly(model, forecast)
                fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
                st.plotly_chart(fig1)

                # st.subheader('Forecast Components')
                # st.write('This plot breaks down the forecast into trend, weekly, and yearly components.')
                # fig2 = plot_components_plotly(model, forecast)
                # fig2.update_traces(line=dict(color='white'))
                # st.plotly_chart(fig2)

                # st.subheader('Performance Metrics')

                # st.write('The metrics below provide a quantitative measure of the model’s accuracy. The Mean Absolute Error (MAE) is the average absolute difference between predicted and actual values, Mean Squared Error (MSE) is the average squared difference, and Root Mean Squared Error (RMSE) is the square root of MSE, which is more interpretable in the same units as the target variable.')

                # actual = df['y']
                # predicted = forecast['yhat'][:len(df)]
                # metrics = calculate_performance_metrics(actual, predicted)
                # st.metric(label="Mean Absolute Error (MAE)", value="{:.2f}".format(metrics['MAE']), delta="Lower is better")
                # st.metric(label="Mean Squared Error (MSE)", value="{:.2f}".format(metrics['MSE']), delta="Lower is better")
                # st.metric(label="Root Mean Squared Error (RMSE)", value="{:.2f}".format(metrics['RMSE']), delta="Lower is better")

# Run the main function
if __name__ == "__main__":
    main()
