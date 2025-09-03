# btc_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
import yfinance as yf
import requests
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC Predictive Dashboard", layout="wide")

# ---------------------------
# 1. Initialize APIs
# ---------------------------
cg = CoinGeckoAPI()

# ---------------------------
# 2. Helper Functions
# ---------------------------
def get_crypto_history(coin_id, vs_currency='usd', days=180):
    vs_currency = vs_currency.lower()
    supported = cg.get_supported_vs_currencies()
    if vs_currency not in supported:
        raise ValueError(
            f"Invalid vs_currency: '{vs_currency}'. Supported currencies: {supported[:10]} ..."
        )

    data = cg.get_coin_market_chart_by_id(
        id=coin_id, vs_currency=vs_currency, days=days
    )
    df = pd.DataFrame(data['prices'], columns=['timestamp', coin_id])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def get_macro_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)["Close"]
    df = df.to_frame(name=ticker)
    return df

def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=180"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df['value'] = df['value'].astype(float)
    return df[['value']]

# ---------------------------
# 3. Fetch Data
# ---------------------------
days = st.sidebar.slider("Days of Data", 30, 365, 180)

btc = get_crypto_history('bitcoin', vs_currency='usd', days=days)
eth = get_crypto_history('ethereum', vs_currency='usd', days=days)
sol = get_crypto_history('solana', vs_currency='usd', days=days)

start = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
end = datetime.today().strftime('%Y-%m-%d')
dxy = get_macro_data('DX-Y.NYB', start, end)
gold = get_macro_data('GC=F', start, end)
sp500 = get_macro_data('^GSPC', start, end)
fng = get_fear_greed_index()

df = pd.concat([btc, eth, sol, dxy, gold, sp500, fng], axis=1)
df.dropna(inplace=True)
df.columns = ['BTC', 'ETH', 'SOL', 'DXY', 'GOLD', 'SP500', 'FNG']

st.title("BTC Predictive Model Dashboard")
st.subheader("Head of Data")
st.dataframe(df.tail(10))

# ---------------------------
# 4. Correlation Analysis
# ---------------------------
corr_matrix = df.corr()
st.subheader("Correlation Matrix")
st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

# ---------------------------
# 5. Feature Selection
# ---------------------------
top_features = corr_matrix['BTC'][abs(corr_matrix['BTC']) > 0.3].index.drop('BTC')
X = df[top_features]
y = df['BTC']

# ---------------------------
# 6. Regression Model
# ---------------------------
model = LinearRegression()
model.fit(X, y)
df['BTC_pred'] = model.predict(X)

st.subheader("Regression Coefficients")
coef_df = pd.DataFrame({
    "Feature": top_features,
    "Coefficient": model.coef_
})
st.dataframe(coef_df)
st.write(f"Intercept: {model.intercept_:.4f}")

# ---------------------------
# 7. Visualizations
# ---------------------------
st.subheader("BTC Actual vs Predicted")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['BTC'], label='Actual BTC')
ax.plot(df.index, df['BTC_pred'], label='Predicted BTC', linestyle='--')
ax.legend()
st.pyplot(fig)

st.subheader("Pair Plots (BTC vs Top Features)")
sns.pairplot(df, vars=['BTC'] + list(top_features))
st.pyplot(plt.gcf())

# ---------------------------
# 8. Optional: Stochastic Simulation
# ---------------------------
st.subheader("Stochastic Simulation Example (20% uncertainty)")
volatility = df['BTC'].pct_change().std()
simulated = df['BTC_pred'] + np.random.normal(0, volatility, size=len(df))
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.index, df['BTC_pred'], label='Deterministic Prediction')
ax2.plot(df.index, simulated, label='Simulated with Random Noise', linestyle='--')
ax2.legend()
st.pyplot(fig2)
