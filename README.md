# Personalized-Robo-Advisory-with-Alternative-Data

# Alt-Data Robo Advisor

# Personalized-Robo-Advisory-with-Alternative-Data  
# Alt-Data Robo Advisor  

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-orange)](https://streamlit.io/)  
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-lightblue)](https://pandas.pydata.org/)  
[![NumPy](https://img.shields.io/badge/NumPy-2.4.1-yellow)](https://numpy.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-green)](https://scikit-learn.org/)  
[![yfinance](https://img.shields.io/badge/yfinance-latest-lightgrey)](https://pypi.org/project/yfinance/)  
[![NLTK](https://img.shields.io/badge/NLTK-NLP-red)](https://www.nltk.org/)  
[![spaCy](https://img.shields.io/badge/spaCy-NLP-teal)](https://spacy.io/)  
[![TF--IDF](https://img.shields.io/badge/TF--IDF-Text%20Vectorization-purple)](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)  
[![Cosine Similarity](https://img.shields.io/badge/Cosine-Similarity-pink)](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity)  
[![Pearson](https://img.shields.io/badge/Pearson-Correlation-brightgreen)](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)  
[![Spearman](https://img.shields.io/badge/Spearman-Correlation-blueviolet)](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)  
[![Rolling Windows](https://img.shields.io/badge/Rolling-Windows-brown)](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)  
[![API Ingestion](https://img.shields.io/badge/API-Data%20Ingestion-lightseagreen)](https://pypi.org/project/yfinance/)  
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-Ratio-gold)](https://en.wikipedia.org/wiki/Sharpe_ratio)  
[![Returns](https://img.shields.io/badge/Financial-Returns-darkorange)](https://www.investopedia.com/terms/r/return.asp)  


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
## Tech Stack & Methods

**Core Stack**
- Python 3.11  
- Streamlit (Interactive Web App)  
- Pandas & NumPy (Data Wrangling)  
- scikit-learn (Logistic Regression, TF-IDF, Cosine Similarity)  
- yfinance (Market Data API)

**Quant & ML Techniques**
- Logistic Regression for signal classification  
- TF-IDF vectorization for text sentiment features  
- Cosine similarity for sentiment relevance scoring  
- Pearson & Spearman correlation for feature relevance  
- Rolling windows for volatility & momentum  
- Probabilistic confidence scoring  

**NLP & Alternative Data**
- Twitter sentiment ingestion  
- Google Trends signals  
- NLTK & spaCy preprocessing pipeline  
- Keyword scoring & aggregation  

**Portfolio Analytics**
- Daily returns computation  
- Sharpe ratio performance metric  
- Equity curve simulation  
- Signal backtesting framework  

**Deployment**
- Streamlit Cloud  
- GitHub version control  
- Modular project architecture  

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


