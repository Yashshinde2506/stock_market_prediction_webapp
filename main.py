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
    # background_image = """
    # <style>
    # [data-testid="stAppViewContainer"] > .main {
    #     background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    #     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    #     background-position: center;  
    #     background-repeat: no-repeat;
    # }
    # </style>
    # """

    # st.markdown(background_image, unsafe_allow_html=True)

    
    st.sidebar.header('Navigation')
    page = st.sidebar.radio("Pick one", ["Home", "User Login"])

    if page=="Home":
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
            st.title('Login')
            user1 = st.text_input('Username')
            pass1 = st.text_input('Password', type='password')
            submit = st.button('Submit')
            if submit:
                if user1!='Yash' and pass1!='Pass123':
                    st.error("Wrong Password")
                elif user1=='Yash' and pass1=='Pass123':
                    st.success('Successfully Login')

                    st.sidebar.header('User Input Parameters')
                    ticker_symbol = st.sidebar.text_input('Enter Ticker Symbol', 'RACE')
                    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
                    end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))
                    forecast_horizon = st.sidebar.selectbox('Forecast Horizon', options=['1 year', '2 years', '3 years', '5 years'], format_func=lambda x: x.capitalize())

            # Introduction
            # Set up the layout
                    # Convert the selected horizon to days
                    horizon_mapping = {'1 year': 365, '2 years': 730, '3 years': 1095, '5 years': 1825}
                    forecast_days = horizon_mapping[forecast_horizon]

                    if st.sidebar.button('Forecast Stock Prices'):
                        with st.spinner('Fetching data...'):
                            df = fetch_stock_data(ticker_symbol, start_date, end_date)

                        with st.spinner('Training model...'):
                            model = train_prophet_model(df)
                            forecast = make_forecast(model, forecast_days)

                        st.subheader('Forecast Data')
                        st.write('The table below shows the forecasted stock prices along with the lower and upper bounds of the predictions.')
                        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

                        st.subheader('Forecast Plot')
                        st.write('The plot below visualizes the predicted stock prices with their confidence intervals.')
                        fig1 = plot_plotly(model, forecast)
                        fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
                        st.plotly_chart(fig1)

                        st.subheader('Forecast Components')
                        st.write('This plot breaks down the forecast into trend, weekly, and yearly components.')
                        fig2 = plot_components_plotly(model, forecast)
                        fig2.update_traces(line=dict(color='white'))
                        st.plotly_chart(fig2)

                        st.subheader('Performance Metrics')

                        st.write('The metrics below provide a quantitative measure of the model’s accuracy. The Mean Absolute Error (MAE) is the average absolute difference between predicted and actual values, Mean Squared Error (MSE) is the average squared difference, and Root Mean Squared Error (RMSE) is the square root of MSE, which is more interpretable in the same units as the target variable.')

                        actual = df['y']
                        predicted = forecast['yhat'][:len(df)]
                        metrics = calculate_performance_metrics(actual, predicted)
                        st.metric(label="Mean Absolute Error (MAE)", value="{:.2f}".format(metrics['MAE']), delta="Lower is better")
                        st.metric(label="Mean Squared Error (MSE)", value="{:.2f}".format(metrics['MSE']), delta="Lower is better")
                        st.metric(label="Root Mean Squared Error (RMSE)", value="{:.2f}".format(metrics['RMSE']), delta="Lower is better")

# Run the main function
if __name__ == "__main__":
    main()
