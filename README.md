# Personalized-Robo-Advisory-with-Alternative-Data

# Alt-Data Robo Advisor

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-orange)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-lightblue)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.1-yellow)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-green)](https://scikit-learn.org/)
[![yfinance](https://img.shields.io/badge/yfinance-latest-lightgrey)](https://pypi.org/project/yfinance/)
[![TF-IDF](https://img.shields.io/badge/TF--IDF-lightgreen)](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf)
[![Cosine Similarity](https://img.shields.io/badge/Cosine_Similarity-purple)](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity)
[![Pearson & Spearman](https://img.shields.io/badge/Pearson%2FSpearman-red)](https://docs.scipy.org/doc/scipy/reference/stats.html)
[![Rolling Windows](https://img.shields.io/badge/Rolling_Windows-cyan)](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
[![NLP (NLTK / spaCy)](https://img.shields.io/badge/NLP-NLTK%2FspaCy-orange)](https://www.nltk.org/)
[![API Data Ingestion](https://img.shields.io/badge/API-Ingestion-blueviolet)](https://requests.readthedocs.io/)
[![Financial Metrics](https://img.shields.io/badge/Financial_Metrics-lightgrey)](https://www.investopedia.com/terms/s/sharperatio.asp)

---

## Overview

**Alt-Data Robo Advisor** is a multi-ticker trading signal generator and portfolio backtester powered by Python. It combines historical price data and technical features to generate **BUY / SELL / HOLD signals** for selected tickers, with confidence scores.  

Personalized-Robo-Advisory-with-Alternative-Data/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── signal_model.py
│   ├── signal_generator.py
│   └── backtester.py

The app includes:

- Multi-ticker support (AAPL, MSFT, GOOG, TSLA, NVDA…)
- Live feature generation for signals
- Logistic Regression-based signal prediction
- Portfolio backtesting and equity curve visualization
- Streamlit interface for interactive analysis
- CSV export for signals and portfolio equity curve

---

## Features

- **Live Features Table:** View calculated indicators like 5-day returns, 10-day volatility, and sentiment metrics.
- **Signal Prediction:** Logistic Regression model predicts BUY / SELL signals with confidence.
- **Portfolio Aggregation:** Mean equity curve across all selected tickers.
- **Backtesting:** Evaluate historical performance and Sharpe ratio of generated signals.
- **Streamlit Interface:** Interactive web-based UI to select tickers, view signals, and download CSV files.

---

## Installation

python -m venv venv311
.\venv311\Scripts\activate

pip install -r requirements.txt
streamlit
pandas
numpy
scikit-learn
yfinance

robo_advisor_altdata/
│
├─ src/                # Core Python modules
│  ├─ data_loader.py   # Load historical price data using yfinance
│  ├─ features.py      # Feature engineering
│  ├─ signal_model.py  # Train/load model, predict signals
│  ├─ signal_generator.py # Generate signals
│  ├─ backtester.py    # Backtesting logic
│
├─ models/             # Saved model files
├─ app.py              # Streamlit app
├─ config.py           # Tickers, dates, paths
├─ requirements.txt    # Python dependencies
└─ README.md           # Project documentation


---
License

MIT License © Peterson Muriuki


