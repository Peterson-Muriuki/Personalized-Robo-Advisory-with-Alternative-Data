import pandas as pd
import numpy as np

def backtest(prices, signals_df):
    # Example: naive backtest
    returns = prices.pct_change().fillna(0)
    
    # Align signals with returns
    signals = signals_df["signal"].shift(1).fillna(0)

    equity_curve = (1 + returns.squeeze() * signals).cumprod()

    # Ensure equity_curve is a pandas Series
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve, index=prices.index)

    total_return = equity_curve.iloc[-1] - 1
    sharpe_ratio = equity_curve.pct_change().mean() / equity_curve.pct_change().std() * np.sqrt(252)

    return {
        "equity_curve": equity_curve,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio
    }
