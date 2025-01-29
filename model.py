import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Function to fetch stock data from Alpha Vantage
def get_stock_data(stock_symbol, start_date, end_date):
    API_KEY = "5YU56HI73O1R1OBX"  # Replace with your actual Alpha Vantage API key
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={API_KEY}&outputsize=full"

    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        st.error("Failed to retrieve stock data. Check the symbol or API key.")
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
    df.index = pd.to_datetime(df.index)
    df = df.loc[start_date:end_date]

    # Ensure data is not empty
    if df.empty:
        st.error("No data available for the selected stock and date range.")
        return None

    return df[::-1]

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Stock Prediction by Saideep Kasipathy", layout="wide")
    st.sidebar.title("Stock Prediction App")
    st.sidebar.markdown("**Created by [Saideep Kasipathy](https://www.linkedin.com/in/sdk4/)**")
    st.sidebar.info("Enter a stock symbol and explore the market trends and future predictions.")

    # Input for stock symbol
    stock_symbol = st.sidebar.text_input("Enter a Stock Symbol", value="AAPL").upper()

    # Date range selection
    today = datetime.today()
    start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", today)

    df = get_stock_data(stock_symbol, start_date, end_date)

    if df is not None:
        # Drop rows with missing data
        df = df.dropna(subset=["Close"])

        if df.empty:
            st.error("No valid data points found after removing NaN values.")
            return

        # Display stock data
        st.header(f"Stock Data for {stock_symbol}")
        st.dataframe(df.tail(10))

        # Plot stock price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price", line=dict(color="royalblue")))
        fig.update_layout(title=f"Stock Closing Price of {stock_symbol}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Model selection and prediction
        st.header("Stock Price Prediction")
        model_choice = st.selectbox("Choose a prediction model", ["Ridge Regression", "Decision Tree", "SVR", "Gradient Boosting"])
        prediction_days = st.slider("Days to Forecast", min_value=1, max_value=30, value=7)

        # Prepare data for training
        df["Target"] = df["Close"].shift(-prediction_days)
        features = df[["Close"]].values

        # Ensure features are valid for scaling
        if len(features) == 0:
            st.error("No valid data points available for scaling.")
            return

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        X = features_scaled[:-prediction_days]
        y = df["Target"].dropna().values

        # Ensure there is enough data for training
        if len(X) == 0 or len(y) == 0:
            st.error("Not enough data points for training. Adjust the date range or prediction days.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train selected model
        models = {
            "Ridge Regression": Ridge(),
            "Decision Tree": DecisionTreeRegressor(),
            "SVR": SVR(),
            "Gradient Boosting": GradientBoostingRegressor()
        }
        model = models[model_choice]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate model
        st.write(f"**Model: {model_choice}**")
        st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")

        # Forecast future prices
        future_features = features_scaled[-prediction_days:]
        forecast = model.predict(future_features)

        # Display predictions
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
        st.write("**Predicted Stock Prices:**")
        st.dataframe(forecast_df)

        # Plot predictions
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Historical Close Price", line=dict(color="gray")))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines+markers", name="Predicted Price", line=dict(color="red")))
        fig_pred.update_layout(title=f"Predicted Stock Prices for {stock_symbol}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_pred, use_container_width=True)

# Ensure the script runs properly when executed
if __name__ == "__main__":
    main()



