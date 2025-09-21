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
data_load_state = st.text("Data Loaded")

st.subheader("Raw Data For Last 4 Days")
st.write(data.tail())

def plot_data(data, select_stock):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[('Date', '')], y=data[('Open',select_stock)], name='stock_open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data[('Date', '')], y=data[('Close', select_stock)], name='stock_close', line=dict(color='red')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_data(data, select_stock)

data.columns = [col[0] for col in data.columns]

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})


model = Prophet()
model.fit(df_train)
future_data = model.make_future_dataframe(periods=period)
forecast = model.predict(future_data)

st.subheader(f"Forecast Data for {select_stock}: ")
st.write(forecast.tail())

st.write("Forecast Chart For Predicted Stock Price")
future_chart = plot_plotly(model, forecast)
st.plotly_chart(future_chart)

st.write("Forecast Components:")
components_future = model.plot_components(forecast)
st.write(components_future)

from prophet.diagnostics import cross_validation, performance_metrics

data_load_state = st.text("Calculating Model Accuracy.....")
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
df_metrics = performance_metrics(df_cv)
accuracy = (1 - df_metrics['mape'].mean()) * 100

data_load_state.text("Calculated!!")
st.subheader(f"Model accuracy: {accuracy:.2f}%")


## Available columns: [('Date', ''), ('Close', '^GSPC'), ('High', '^GSPC'), ('Low', '^GSPC'), ('Open', '^GSPC'), ('Volume', '^GSPC')]