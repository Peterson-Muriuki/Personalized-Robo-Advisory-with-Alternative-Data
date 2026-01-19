# app.py

import os
import sys

# Ensure 'src' folder is in Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from config import TICKERS, START_DATE, END_DATE, MODEL_PATH
from data_loader import load_price_data
from features import build_features
from signal_model import train_model, load_model, predict_signal
from signal_generator import generate_signals
from backtester import backtest

st.set_page_config(page_title="Alt-Data Robo Advisor", layout="wide")
st.title("Alt-Data Robo Advisor")
st.markdown("Analyze multiple tickers, generate signals, and visualize portfolio performance.")

# -----------------------------
# Sidebar controls
# -----------------------------
selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=TICKERS,
    default=TICKERS
)

show_historical = st.sidebar.checkbox("Show Historical Prices", value=True)
show_portfolio = st.sidebar.checkbox("Show Portfolio Curve", value=True)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return load_price_data(TICKERS, START_DATE, END_DATE)

price_df = load_data()

if show_historical:
    st.subheader("Historical Prices (last 5 rows)")
    st.dataframe(price_df.tail())

# -----------------------------
# Processing selected tickers
# -----------------------------
all_signals = []
portfolio_returns = []

for ticker in selected_tickers:
    st.subheader(f"Processing {ticker}...")

    prices = price_df[[ticker]]
    sentiment_df = pd.DataFrame()  # placeholder for future sentiment data

    # Build live features
    X_live = build_features(prices, sentiment_df)
    st.write("Live Features")
    st.dataframe(X_live)

    # Load or train model
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model()
        except Exception as e:
            st.warning(f"Failed to load model ({e}). Retraining.")

    if model is None:
        st.info("Training new model...")
        X_hist_list = []
        for i in range(len(prices) - 1):  # leave last row for live prediction
            hist_slice = prices.iloc[:i + 1]
            X_hist_list.append(build_features(hist_slice, sentiment_df))
        X_hist = pd.concat(X_hist_list, ignore_index=True)

        # Simple trend labels (next day return positive -> 1 else 0)
        y_hist = (prices.pct_change().shift(-1) > 0).astype(int).iloc[:-1][ticker]

        # Align lengths
        X_hist = X_hist.iloc[:-1].reset_index(drop=True)
        y_hist = y_hist.reset_index(drop=True)

        model = train_model(X_hist, y_hist)

    # Predict live signal
    signal, confidence = predict_signal(model, X_live)
    st.write(f"**Live Signal:** {signal} (confidence={confidence:.2f})")

    # Generate historical signals and backtest
    signals_df = generate_signals(prices)
    perf = backtest(prices, signals_df)

    if isinstance(perf["equity_curve"], (pd.DataFrame, pd.Series)):
        equity_curve = perf["equity_curve"]
    else:
        equity_curve = pd.Series(perf["equity_curve"])

    all_signals.append({
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "total_return": perf["total_return"],
        "sharpe": perf["sharpe_ratio"]
    })
    portfolio_returns.append(equity_curve)

# -----------------------------
# Portfolio aggregation
# -----------------------------
results_df = pd.DataFrame(all_signals)
results_df["rank"] = results_df["confidence"].rank(ascending=False)

st.subheader("Portfolio Signals")
st.dataframe(results_df.sort_values("rank"))

if show_portfolio and portfolio_returns:
    portfolio_curve = pd.concat(portfolio_returns, axis=1).mean(axis=1)
    st.subheader("Portfolio Equity Curve")
    st.line_chart(portfolio_curve)

# -----------------------------
# Optional CSV export
# -----------------------------
st.download_button(
    label="Download Portfolio Signals CSV",
    data=results_df.to_csv(index=False),
    file_name="portfolio_signals.csv",
    mime="text/csv"
)
