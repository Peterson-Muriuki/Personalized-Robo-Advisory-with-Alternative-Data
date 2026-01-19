import pandas as pd

def generate_signals(price_df):
    short_ma = price_df.rolling(5).mean()
    long_ma = price_df.rolling(20).mean()

    signal = (short_ma > long_ma).astype(int).fillna(0)
    return pd.DataFrame({"signal": signal.values.flatten()}, index=price_df.index)
