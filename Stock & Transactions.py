# — หน้า 1: ราคา + บันทึกธุรกรรม (หุ้น/เงินสด)
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta
from db_utils import (
    init_db, ensure_portfolio, safe_add_trade, read_trades_df, upsert_prices,
    read_cash_ledger, add_cash, compute_cash_balance
)

st.set_page_config(page_title="Stock & Transactions", layout="wide")
st.title("Stock Trend")

# ---------------- Sidebar ----------------
raw_ticker = st.sidebar.text_input("Ticker", "AAPL")
ticker = raw_ticker.split(",")[0].strip().upper()
start = st.sidebar.date_input("Start", date.today() - timedelta(days=365))
end = st.sidebar.date_input("End", date.today())
ma_window = int(st.sidebar.number_input("SMA Window", 5, 300, 20))
portfolio_key = st.sidebar.text_input("Portfolio Key", "portfolio1")

# ---------------- DB init ----------------
init_db()
pf_id = ensure_portfolio(portfolio_key)

# ---------------- Fetch ----------------
@st.cache_data(show_spinner=False)
def fetch_yf(tk: str, start_d: date, end_d: date):
    df = yf.download(tk, start=start_d, end=end_d, auto_adjust=True, progress=False)
    return df if df is not None else pd.DataFrame()

raw = fetch_yf(ticker, start, end)
if raw.empty:
    st.error("ไม่พบข้อมูลจาก yfinance")
    st.stop()

# ---------------- Normalize 'Close' ----------------
def normalize_close_column(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):
        cols = [str(c).strip() for c in df.columns]
        df = df.copy(); df.columns = [c.title() for c in cols]
        for cand in ["Close", "Adj Close", "Adj_Close", "Price"]:
            if cand in df.columns:
                return df[[cand]].rename(columns={cand: "Close"})
        close_like = [c for c in df.columns if "close" in c.lower()]
        if close_like:
            return df[[close_like[0]]].rename(columns={close_like[0]: "Close"})
        return pd.DataFrame()
    cols = df.columns
    if symbol_hint and (symbol_hint, "Close") in cols:
        return df[(symbol_hint, "Close")].to_frame(name="Close")
    if symbol_hint and (symbol_hint, "Adj Close") in cols:
        return df[(symbol_hint, "Adj Close")].to_frame(name="Close")
    candidates = [t for t in cols if ("close" in str(t[-1]).lower()) or (str(t[-1]).lower() == "price")]
    if candidates:
        chosen = None
        for t in candidates:
            if str(t[-1]).lower() == "close": chosen = t; break
        if chosen is None:
            for t in candidates:
                if "adj" in str(t[-1]).lower() and "close" in str(t[-1]).lower(): chosen = t; break
        if chosen is None: chosen = candidates[0]
        return df[chosen].to_frame(name="Close")
    flat_cols = ["::".join(map(str, t)) for t in cols]
    tmp = df.copy(); tmp.columns = flat_cols
    close_like = [c for c in tmp.columns if "close" in c.lower() or c.lower().endswith("price")]
    if close_like:
        return tmp[[close_like[0]]].rename(columns={close_like[0]: "Close"})
    return pd.DataFrame()

df = normalize_close_column(raw, ticker)
if df.empty or "Close" not in df.columns:
    st.error(f"ไม่พบคอลัมน์ Close/Adj Close ใน {ticker}\nคอลัมน์ที่เจอ: {list(raw.columns)}")
    st.stop()

# ---------------- Clean/Transform ----------------
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "Date"
df = df[~df.index.duplicated(keep="last")]
try: df = df.asfreq("B")
except ValueError: pass
df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill()
df = df.dropna(subset=["Close"])
win = max(1, min(ma_window, len(df)))
df["Return"] = df["Close"].pct_change()
df[f"SMA{win}"] = df["Close"].rolling(win).mean()
df["EMA20"] = df["Close"].ewm(span=min(20, max(2, len(df))), adjust=False).mean()

# ---------------- Persist price cache ----------------
upsert_prices(ticker, df.reset_index()[["Date", "Close"]])

# ---------------- Layout ----------------
colL, colR = st.columns([2, 1])

# ==== Left: Price & trades ====
with colL:
    st.subheader(f"📈 {ticker} — Price & MAs")
    fig = px.line(df.reset_index(), x="Date", y=["Close", f"SMA{win}", "EMA20"], title=f"{ticker} Price")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Data tail 60")
    st.dataframe(df.tail(60), use_container_width=True)

    # --- Trade form (BUY/SELL) ---
    st.divider()
    st.subheader("🧾 เพิ่มธุรกรรมหุ้น (BUY/SELL)")
    import datetime as dt
    with st.form("trade_form", clear_on_submit=True):
        c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1, 1, 1, 1, 1])
        d   = c1.date_input("วันที่", dt.date.today())
        sym = c2.text_input("สัญลักษณ์", ticker).upper().strip()
        side= c3.selectbox("ประเภท", ["BUY", "SELL"])
        qty = c4.number_input("จำนวน", min_value=1, value=10)
        price_input = c5.number_input("ราคา/หน่วย", min_value=0.0, value=float(df["Close"].iloc[-1]))
        fee = c6.number_input("ค่าธรรมเนียม", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("บันทึกธุรกรรมหุ้น")
        if submitted:
            msg = safe_add_trade(pf_id, sym, d.strftime("%Y-%m-%d"), side, int(qty), float(price_input), float(fee))
            if msg.startswith("✅"):
                st.success(msg)
            else:
                st.error(msg)

    st.subheader("รายการธุรกรรมหุ้นล่าสุด")
    trades = read_trades_df(pf_id)
    st.dataframe(trades, use_container_width=True)

# ==== Right: Cash ledger ====
with colR:
    st.subheader("💰Cash Management")
    # Cash balance (live)
    cash_now = compute_cash_balance(pf_id)
    st.metric("Cash Balance", f"{cash_now:,.2f}")

    # Cash form
    with st.form("cash_form", clear_on_submit=True):
        c1, c2 = st.columns([1, 1])
        cash_date = c1.date_input("วันที่ (เงินสด)", date.today())
        cash_type = c2.selectbox("ประเภท", ["DEPOSIT", "WITHDRAW", "FEE", "ADJUST"])
        amount = st.number_input("จำนวนเงิน (USD)", min_value=0.0, value=500.0)
        note = st.text_input("หมายเหตุ (ถ้ามี)", "")
        ok = st.form_submit_button("บันทึก")
        if ok:
            # บันทึกด้วยจำนวนบวกเสมอ ฟังก์ชันจะจัดสัญญะตาม type ให้เอง
            add_cash(pf_id, cash_date.strftime("%Y-%m-%d"), cash_type, float(amount), note or None)
            st.success("✅ บันทึกแล้ว")

    st.caption("การเปลี่ยนแปลงล่าสุด")
    cash_df = read_cash_ledger(pf_id)
    st.dataframe(cash_df.tail(20), use_container_width=True)
