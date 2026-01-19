from config import TICKERS, START_DATE, END_DATE, MODEL_PATH
from data_loader import load_price_data
from features import build_features
from signal_model import train_model, load_model, predict_signal
from signal_generator import generate_signals
from backtester import backtest
import pandas as pd
import os

print("Running Alt-Data Robo Advisor...")

# Load historical prices for all tickers
price_df = load_price_data(TICKERS, START_DATE, END_DATE)

all_signals = []
portfolio_returns = []

for ticker in TICKERS:
    print(f"\nProcessing {ticker}...")

    # Slice prices for this ticker
    prices = price_df[[ticker]]

    # Placeholder for sentiment — replace with actual sentiment DataFrame
    sentiment_df = pd.DataFrame()  

    # Build live features (most recent row)
    X_live = build_features(prices, sentiment_df)
    print("Live Features:")
    print(X_live)

    # Attempt to load existing model
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model()
        except Exception as e:
            print(f"Warning: failed to load model ({e}). Will retrain.")

    # If model missing or corrupted, train a new one
    if model is None:
        print("Training new model...")

        # Build historical features for training
        X_hist_list = []
        for i in range(len(prices) - 1):  # exclude last row for live prediction
            hist_slice = prices.iloc[:i + 1]
            X_hist_list.append(build_features(hist_slice, sentiment_df))
        X_hist = pd.concat(X_hist_list, ignore_index=True)

        # Create simple trend labels (next day return > 0 -> 1 else 0)
        y_hist = (prices.pct_change().shift(-1) > 0).astype(int).iloc[:-1][ticker]

        # Align X_hist and y_hist lengths
        X_hist = X_hist.iloc[:-1].reset_index(drop=True)
        y_hist = y_hist.reset_index(drop=True)

        model = train_model(X_hist, y_hist)

    # Predict signal for live row
    signal, confidence = predict_signal(model, X_live)

    # Generate signals for the entire series
    signals_df = generate_signals(prices)

    # Backtest multi-ticker-safe: ensure equity_curve is a Series
    perf = backtest(prices, signals_df)
    if isinstance(perf["equity_curve"], pd.DataFrame) or isinstance(perf["equity_curve"], pd.Series):
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

# Aggregate results
results_df = pd.DataFrame(all_signals)
results_df["rank"] = results_df["confidence"].rank(ascending=False)

# Combine portfolio equity curves: equal-weight portfolio
portfolio_curve = pd.concat(portfolio_returns, axis=1).mean(axis=1)

print("\nPortfolio Results")
print(results_df.sort_values("rank"))

print("\nPortfolio Equity Curve (last 5 days)")
print(portfolio_curve.tail())

# Optional: save results to CSV
results_df.to_csv("portfolio_signals.csv", index=False)
portfolio_curve.to_csv("portfolio_equity_curve.csv", index=True)
