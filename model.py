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

# Market Stack API key
API_KEY = "0177c21f33c260c026bc186a1b286d58"

# Function to fetch stock data from Market Stack
def get_stock_data_marketstack(stock_symbol, start_date, end_date):
    base_url = "https://api.marketstack.com/v1/eod"
    params = {
        "access_key": API_KEY,
        "symbols": stock_symbol,
        "date_from": start_date,
        "date_to": end_date,
        "limit": 1000,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Debugging: Print the API response and status code
    st.write(f"API Response Status Code: {response.status_code}")
    st.write(f"API Response: {data}")

    # Check for request limits in headers (if available)
    request_headers = response.headers
    remaining_requests = request_headers.get("X-RateLimit-Remaining", "N/A")
    reset_time = request_headers.get("X-RateLimit-Reset", "N/A")

    # Display request limit info in the sidebar
    st.sidebar.info(f"Remaining Requests: {remaining_requests}")
    st.sidebar.info(f"Reset Time: {reset_time}")

    # Check if the API returned valid data
    if "data" not in data:
        st.error(f"Error: {data.get('error', {}).get('message', 'Unknown error')}")
        return None

    df = pd.DataFrame(data["data"])

    # Select a price column to use
    if 'close' in df.columns:
        df['Price'] = df['close']
    elif 'open' in df.columns:
        st.warning("Using 'Open' price as 'Close' price is unavailable.")
        df['Price'] = df['open']
    else:
        st.error("Neither 'Close' nor 'Open' price columns are available in the data.")
        return None

    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Increment request counter manually
    increment_request_count()

    return df

# Function to manually track requests (stored in Streamlit session state)
def increment_request_count():
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    st.session_state.request_count += 1

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Stock Prediction by Saideep Kasipathy", layout="wide")
    st.sidebar.title("Stock Prediction App")
    st.sidebar.markdown("**Created by [Saideep Kasipathy](https://www.linkedin.com/in/sdk4/)**")
    st.sidebar.info("Enter a stock symbol and explore the market trends and future predictions.")

    # Display request count in the sidebar
    st.sidebar.info(f"Total Requests Made: {st.session_state.get('request_count', 0)}")

    # Input for stock symbol
    stock_symbol = st.sidebar.text_input("Enter a Stock Symbol", value="AAPL").upper()

    # Date range selection
    today = datetime.today()
    start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365)).isoformat()
    end_date = st.sidebar.date_input("End Date", today).isoformat()

    # Retry button to fetch data again
    if st.sidebar.button("Retry Fetching Data"):
        st.info("Retrying data fetch...")

    # Fetch data using Market Stack
    df = get_stock_data_marketstack(stock_symbol, start_date, end_date)

    if df is not None:
        # Drop rows with missing data in the 'Price' column
        df = df.dropna(subset=["Price"])

        if df.empty:
            st.error("No valid data points found after removing NaN values.")
            return

        # Display stock data
        st.header(f"Stock Data for {stock_symbol}")
        st.dataframe(df.tail(10))

        # Plot stock price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["Price"], mode="lines", name="Price", line=dict(color="royalblue")))
        fig.update_layout(title=f"Stock Price of {stock_symbol}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Model selection and prediction
        st.header("Stock Price Prediction")
        model_choice = st.selectbox("Choose a prediction model", ["Ridge Regression", "Decision Tree", "SVR", "Gradient Boosting"])
        prediction_days = st.slider("Days to Forecast", min_value=1, max_value=30, value=7)

        # Prepare data for training
        df["Target"] = df["Price"].shift(-prediction_days)
        features = df[["Price"]].values

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
        future_dates = [df["date"].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
        st.write("**Predicted Stock Prices:**")
        st.dataframe(forecast_df)

        # Plot predictions
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df["date"], y=df["Price"], mode="lines", name="Historical Price", line=dict(color="gray")))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines+markers", name="Predicted Price", line=dict(color="red")))
        fig_pred.update_layout(title=f"Predicted Stock Prices for {stock_symbol}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_pred, use_container_width=True)

# Ensure the script runs properly when executed
if __name__ == "__main__":
    main()
