import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


starting = "2016-01-01" # Start date for the data
today = date.today().strftime("%Y-%m-%d") # Current date save in string and ymd format

st.subheader(today)

## Title of webpage
st.title("Stock Prediction Application")

## List of stocks
stocks = ("^GSPC", "^DJI", "AAPL", "GOOG", "AMZN", "MSFT")

## Selection box UI
select_stock = st.selectbox("Select Dataset: ", stocks)

## Number of years for prediction slider UI
n_years = st.selectbox("Prediction Years Amount: ", (1, 2, 3, 4))

## Defining period in days
period = n_years * 365

@st.cache_data ## Caching the retrieved data
def load_stock(ticker):
    ## Download all data from start date to today
    data = yf.download(ticker, start=starting, end=today, auto_adjust=True) ## data is pandas dataframe
    data.reset_index(inplace=True) ## Putting date in the very first column
    return data

## Getting the dataframe for selected stock ticker
data = load_stock(select_stock)
data_load_state = st.text("Data Loaded") ## Display once the data is loaded successfully

st.subheader("Raw Data For Last 4 Days")
st.write(data.tail()) ## Checking the dataset

## Function to plot the raw data using plotly
def plot_data(data, select_stock):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[('Date', '')], y=data[('Open',select_stock)], name='stock_open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data[('Date', '')], y=data[('Close', select_stock)], name='stock_close', line=dict(color='red')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

## Running the function
plot_data(data, select_stock)

## In raw data, the columns are in tuple format, here we are changing them into string format
data.columns = [col[0] for col in data.columns]

## Creating dataset for training prophet model
df_train = data[['Date', 'Close']]

## Renaming the columns in train dataset because prophet strictly requires certain names for them
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

## Creating the model and training it
model = Prophet()
model.fit(df_train)
## Creating dataframe with predicted values for the stock in user-specified period
future_data = model.make_future_dataframe(periods=period)
forecast = model.predict(future_data)

## Checking the forcasted data values
st.subheader(f"Forecast Data for {select_stock}: ")
st.write(forecast.tail())

## Creating the chart for visual representation of the predicted stock price
st.write("Forecast Chart For Predicted Stock Price")
future_chart = plot_plotly(model, forecast)
st.plotly_chart(future_chart)

## Creating components (weekly, monthly, 6 months etc) for the predicted data
st.write("Forecast Components:")
components_future = model.plot_components(forecast)
st.write(components_future)


## This section is used to calculate the total accuracy of the model
## This is independent of time frame selected by user for forecast
## It calculates the accuracy by training on initial 730 days of data and re-training every 180 days and forecasts=ing up to 365 days
from prophet.diagnostics import cross_validation, performance_metrics

## Display while calculating the accuracy, because it takes a while to run on prophet
data_load_state = st.text("Calculating Model Accuracy.....")
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days') ## Cross Validation
df_metrics = performance_metrics(df_cv) ## Gives all the performance metrics of the model
accuracy = (1 - df_metrics['mape'].mean()) * 100 ## Calculate the mean of the metrics and represent them in percentage format

data_load_state.text("Calculated!!") ## Displays after the accuracy is calculated
st.subheader(f"Model accuracy: {accuracy:.2f}%") ## Display accuracy itself