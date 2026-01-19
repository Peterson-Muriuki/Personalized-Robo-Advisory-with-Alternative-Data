import yfinance as yf
import pandas as pd

def load_price_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        group_by="ticker"
    )

    if len(tickers) == 1:
        data = data["Close"].to_frame(tickers[0])
    else:
        data = data.xs("Close", level=1, axis=1)

    data = data.dropna()
    return data
