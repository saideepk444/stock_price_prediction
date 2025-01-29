import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error


st.title('Stock Price Predictions')
st.sidebar.info(
    'Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info(
    "Created and designed by [Saideep Kasipathy](https://www.linkedin.com/in/saideepk4/)")


def main():
    option = st.sidebar.selectbox(
        'Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()


@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df


option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %
                           (start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')


data = download_data(option, start_date, end_date)
scaler = StandardScaler()


def tech_indicators():
    st.header('Technical Indicators')

    # Debug: Check if `data` is a DataFrame
    if data is None or not isinstance(data, pd.DataFrame):
        st.error("Error: Data is not a DataFrame. Check Yahoo Finance API response.")
        return

    # Debug: Check if data is empty
    if data.empty:
        st.error("Error: No data was retrieved. Check the stock symbol or date range.")
        return

    # Debug: Print data structure
    st.write("### Sample Data", data.head())  # Show first few rows
    st.write("### Column Names", list(data.columns))  # Show column names
    st.write("### Data Type", type(data))  # Show type of data

    # Ensure 'Close' exists
    if 'Close' not in data.columns:
        st.error("Error: 'Close' column not found in data.")
        return

    # Debug: Check type of `data['Close']`
    st.write("### Type of data['Close']", type(data['Close']))

    # Ensure 'Close' is a valid numeric series
    try:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    except Exception as e:
        st.error(f"Error converting 'Close' column: {e}")
        return

    # Ensure 'Close' is not all NaN
    if data['Close'].isna().all():
        st.error("Error: 'Close' column contains only NaN values.")
        return

    if len(data) < 20:
        st.error("Error: Not enough data to compute Bollinger Bands.")
        return

    # Compute indicators
    bb_indicator = BollingerBands(data['Close'])
    bb = data.copy()
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()

    # Drop NaN values
    bb = bb.dropna(subset=['bb_h', 'bb_l'])

    # MACD
    macd = MACD(data['Close']).macd().dropna()
    # RSI
    rsi = RSIIndicator(data['Close']).rsi().dropna()
    # SMA
    sma = SMAIndicator(data['Close'], window=14).sma_indicator().dropna()
    # EMA
    ema = EMAIndicator(data['Close']).ema_indicator().dropna()

    # Display
    option = st.radio('Choose a Technical Indicator to Visualize', [
                      'Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    if option == 'Close':
        st.line_chart(data['Close'])
    elif option == 'BB':
        st.line_chart(bb[['Close', 'bb_h', 'bb_l']])
    elif option == 'MACD':
        st.line_chart(macd)
    elif option == 'RSI':
        st.line_chart(rsi)
    elif option == 'SMA':
        st.line_chart(sma)
    elif option == 'EMA':
        st.line_chart(ema)





def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))


def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor',
                     'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    # spliting the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()
