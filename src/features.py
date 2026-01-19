import pandas as pd
import numpy as np

def _safe_scalar(x):
    """Convert pandas Series / numpy to a single float safely."""
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return 0.0
        return float(x.iloc[-1])
    try:
        return float(x)
    except Exception:
        return 0.0

def build_features(price_df, sentiment_df=None):
    """
    Build features for one ticker or a multi-ticker scenario.
    
    Parameters:
    - price_df: pd.DataFrame or pd.Series with price data (single or multi-ticker)
    - sentiment_df: pd.DataFrame with 'sentiment' column (optional)
    
    Returns:
    - pd.DataFrame with features: return_5d, volatility_10d, avg_sentiment, sentiment_std
    """

    # Handle single-column DataFrame or Series
    if isinstance(price_df, pd.DataFrame):
        if "Close" in price_df.columns:
            close = price_df["Close"]
        elif price_df.shape[1] == 1:
            close = price_df.iloc[:, 0]
        else:
            # Multi-ticker: take mean of all tickers as a simple aggregation
            close = price_df.mean(axis=1)
    elif isinstance(price_df, pd.Series):
        close = price_df
    else:
        raise ValueError("price_df must be a pd.Series or pd.DataFrame")

    # Calculate returns
    returns = close.pct_change()

    # Features dictionary
    features = {}
    features["return_5d"] = _safe_scalar(returns.iloc[-5:].mean())
    features["volatility_10d"] = _safe_scalar(returns.rolling(10).std().iloc[-1])

    # Sentiment features
    if sentiment_df is not None and "sentiment" in sentiment_df.columns and len(sentiment_df) > 0:
        features["avg_sentiment"] = _safe_scalar(sentiment_df["sentiment"].mean())
        features["sentiment_std"] = _safe_scalar(sentiment_df["sentiment"].std())
    else:
        features["avg_sentiment"] = 0.0
        features["sentiment_std"] = 0.0

    # Replace any NaN or infinite values
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    # Return as a single-row DataFrame
    return pd.DataFrame([features])
