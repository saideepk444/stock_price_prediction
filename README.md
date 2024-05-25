# Stock Price Prediction App

This is a Streamlit-based web application for predicting stock prices using historical data from Yahoo Finance.

## Features
- **Visualize:** Visualize historical stock prices and technical indicators, including Close Price, Bollinger Bands (BB), Moving Average Convergence Divergence (MACD), Relative Strength Indicator (RSI), Simple Moving Average (SMA), and Exponential Moving Average (EMA).
- **Recent Data:** View the most recent stock price data.
- **Predict:** Select various machine learning models to predict future stock prices, such as LinearRegression, RandomForestRegressor, ExtraTreesRegressor, KNeighborsRegressor, and XGBoostRegressor.

### Prerequisites
- Python 3.x
- pip
- git (optional)

### Installation

1. Clone the repository or download the source code:
    ```sh
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. (Optional) Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the App

With the necessary packages installed, you can run the app using:

```sh
streamlit run model.py