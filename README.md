# 📈 Stock Portfolio Tracker with Streamlit & SQLite

## Overview

This project is a **Stock Portfolio Tracker** built with **Streamlit**, **yfinance**, and **SQLite**.
It allows users to:

* Fetch stock prices from Yahoo Finance (via `yfinance`)
* Clean & transform stock data (Close price, SMA, EMA, Returns)
* Record buy/sell transactions and cash deposits/withdrawals
* Store data persistently in SQLite
* Compute **Portfolio NAV, Market Value, Realized & Unrealized P\&L**
* Visualize portfolio insights, equity curves, and machine learning forecasts

---

## Features

1. **Stock Viewer**

   * View stock charts with Close, SMA, EMA
   * Historical prices table (downloadable as CSV)

2. **Transactions & Cash Ledger**

   * Add buy/sell trades with quantity, price, and fees
   * Record cash deposits/withdrawals
   * Validation: cannot buy without enough cash or sell more than holdings

3. **Portfolio Dashboard**

   * Shows current holdings, cash, NAV, MV
   * Realized & Unrealized P\&L
   * Equity growth line chart
   * Pie chart of holdings distribution

4. **Machine Learning Insights (Experimental)**

   * Linear regression to forecast short-term stock trend
   * RMSE evaluation of prediction accuracy

---

## Data Engineering Concepts

This project applies fundamental **ETL (Extract–Transform–Load)** principles:

* **Extract**: fetch OHLCV stock data from Yahoo Finance API
* **Clean**: normalize columns, forward-fill missing values, resample as business days
* **Transform**: compute technical indicators (SMA, EMA, returns)
* **Load**: persist prices and trades in SQLite for reproducibility
* **Compute**: derive NAV, MV, P\&L from raw data
* **Visualize**: present via interactive Streamlit dashboard

---

## 📂 Project Structure

```
stock-app/
│
├── Stock & Transactions.py                   # Streamlit main app (Stock Viewer + Transactions)
├── pages/
│   ├── Portfolio_Dashboard.py  # Page 2: Portfolio overview & visualization
│   ├── Analysis_ML_Insights.py # Page 3: ML-based forecasting
│
├── db_utils.py              # SQLite helpers (init DB, trades, cash, prices)
├── portfolio.db             # SQLite database (auto-created)
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## ⚙️ Installation

1. **Clone repository**

```bash
git clone https://github.com/yourusername/stock-app.git
cd stock-app
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run app**

```bash
streamlit run app.py
```

---

## 📦 Requirements

* Python 3.9+
* Libraries:

  * `streamlit`
  * `yfinance`
  * `plotly`
  * `pandas`
  * `numpy`
  * `scikit-learn`

---

## 📊 Example Screenshots

* Stock chart with SMA/EMA overlay
* Portfolio metrics (NAV, P\&L)
* Equity growth curve
* Pie chart of holdings distribution

*(Insert screenshots here after running the app)*

---

## 🌱 Future Improvements

* Add support for multiple portfolios with user login
* Implement more ML models (LSTM for time series)
* Integrate real-time stock quotes
* Deploy to **Streamlit Community Cloud**
