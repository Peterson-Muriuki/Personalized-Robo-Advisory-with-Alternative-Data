import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_yahoo_finance_news(ticker="AAPL", limit=10):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    headlines = []
    for item in soup.select("h3")[:limit]:
        text = item.get_text(strip=True)
        headlines.append(text)

    df = pd.DataFrame({"headline": headlines})
    return df

if __name__ == "__main__":
    df = scrape_yahoo_finance_news("AAPL")
    print(df)
