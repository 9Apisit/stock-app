# pages/2_Portfolio_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from db_utils import init_db, read_all_portfolios, ensure_portfolio, read_trades_df, read_symbols_for_portfolio, average_cost_position, load_prices_df, upsert_prices

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("หน้า 2 — Portfolio Dashboard")

init_db()
pf_list = read_all_portfolios()
if pf_list.empty:
    st.info("ยังไม่มีพอร์ตในระบบ")
    st.stop()

portfolio_key = st.sidebar.selectbox("เลือก Portfolio", options=pf_list["name"].tolist(), index=0)
pf_id = ensure_portfolio(portfolio_key)

trades = read_trades_df(pf_id)
if trades.empty:
    st.info("พอร์ตนี้ยังไม่มีธุรกรรม")
    st.stop()

symbols = read_symbols_for_portfolio(pf_id)
rows = []
for sym in symbols:
    t_df = trades[trades["symbol"] == sym].copy()
    qty, avg_cost, realized = average_cost_position(t_df)
    if qty <= 0:
        continue
    prices = load_prices_df(sym)
    last_price = None
    if not prices.empty:
        last_price = float(prices["Close"].iloc[-1])
    if last_price is None:
        try:
            latest = yf.Ticker(sym).history(period="5d", auto_adjust=True)
            latest.columns = [str(c).title() for c in latest.columns]
            if not latest.empty:
                last_price = float(latest["Close"].iloc[-1])
                upsert_prices(sym, latest.reset_index()[["Date", "Close"]])
        except Exception:
            pass
    if last_price is None:
        last_price = avg_cost
    market_value = qty * last_price
    unreal = (last_price - avg_cost) * qty
    rows.append({"Symbol": sym, "Qty": qty, "AvgCost": avg_cost, "Last": last_price, "MarketValue": market_value, "UnrealizedPnL": unreal, "RealizedPnL": realized})

summary = pd.DataFrame(rows).sort_values("MarketValue", ascending=False)
if summary.empty:
    st.info("ไม่มีสถานะคงเหลือ")
    st.stop()

fig = px.pie(summary, names="Symbol", values="MarketValue", title="Portfolio Allocation")
st.plotly_chart(fig, use_container_width=True)

st.subheader("สรุปสถานะ")
st.dataframe(summary)
