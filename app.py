# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta
from db_utils import init_db, ensure_portfolio, safe_add_trade, read_trades_df, upsert_prices

st.set_page_config(page_title="Stock Viewer & Trade", layout="wide")
st.title("หน้า 1 — ดูกราฟหุ้น + บันทึกธุรกรรม")

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

# ---------------- Normalize 'Close' (กันทุกเคส) ----------------
def normalize_close_column(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    """
    คืน DataFrame ที่มีคอลัมน์เดียวชื่อ 'Close'
    รองรับทั้ง single-index และ MultiIndex (เช่น ('AAPL','Close')) โดยไม่ error
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # เคสคอลัมน์ธรรมดา
    if not isinstance(df.columns, pd.MultiIndex):
        cols = [str(c).strip() for c in df.columns]
        df = df.copy()
        df.columns = [c.title() for c in cols]
        # จับคู่ชื่อที่พบบ่อยก่อน
        for cand in ["Close", "Adj Close", "Adj_Close", "Price"]:
            if cand in df.columns:
                return df[[cand]].rename(columns={cand: "Close"})
        # เผื่อชื่อประหลาดแต่มีคำว่า close
        close_like = [c for c in df.columns if "close" in c.lower()]
        if close_like:
            return df[[close_like[0]]].rename(columns={close_like[0]: "Close"})
        return pd.DataFrame()

    # เคส MultiIndex (เช่นเมื่อดึงหลายสัญลักษณ์)
    cols = df.columns
    if symbol_hint:
        if (symbol_hint, "Close") in cols:
            return df[(symbol_hint, "Close")].to_frame(name="Close")
        if (symbol_hint, "Adj Close") in cols:
            return df[(symbol_hint, "Adj Close")].to_frame(name="Close")

    # หา field ที่ท้ายเลเวลมีคำว่า close/price
    candidates = []
    for tup in cols:
        label = str(tup[-1]).lower()
        if ("close" in label) or (label == "price"):
            candidates.append(tup)
    if candidates:
        # พยายามเลือก 'Close' ก่อน ถ้าไม่มีค่อยใช้ตัวแรก
        chosen = None
        for tup in candidates:
            if str(tup[-1]).lower() == "close":
                chosen = tup; break
        if chosen is None:
            # รองลงมา 'Adj Close'
            for tup in candidates:
                if "adj" in str(tup[-1]).lower() and "close" in str(tup[-1]).lower():
                    chosen = tup; break
        if chosen is None:
            chosen = candidates[0]
        return df[chosen].to_frame(name="Close")

    # ทางสุดท้าย: flatten ชื่อคอลัมน์แล้วจับคำว่า close
    flat_cols = ["::".join(map(str, t)) for t in cols]
    tmp = df.copy()
    tmp.columns = flat_cols
    close_like = [c for c in tmp.columns if "close" in c.lower() or c.lower().endswith("price")]
    if close_like:
        return tmp[[close_like[0]]].rename(columns={close_like[0]: "Close"})
    return pd.DataFrame()

# ใช้ normalize แทนการอ้าง "Close" ตรง ๆ
df = normalize_close_column(raw, ticker)
if df.empty or "Close" not in df.columns:
    st.error(
        "ไม่พบคอลัมน์ราคาหลัก ('Close' หรือ 'Adj Close') ในข้อมูลที่ดึงมา\n"
        f"คอลัมน์ดิบที่เจอ: {list(raw.columns)}"
    )
    st.stop()

# ---------------- Clean ----------------
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "Date"
df = df[~df.index.duplicated(keep="last")]
try:
    df = df.asfreq("B")  # ตั้งเป็นวันทำการ
except ValueError:
    pass
df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill()
df = df.dropna(subset=["Close"])

# ---------------- Transform ----------------
win = max(1, min(ma_window, len(df)))
df["Return"] = df["Close"].pct_change()
df[f"SMA{win}"] = df["Close"].rolling(win).mean()
df["EMA20"] = df["Close"].ewm(span=min(20, max(2, len(df))), adjust=False).mean()

# ---------------- Persist (ELT: เก็บราคาที่ clean ลง SQLite) ----------------
upsert_prices(ticker, df.reset_index()[["Date", "Close"]])

# ---------------- Viz & Table ----------------
fig = px.line(
    df.reset_index(), x="Date", y=["Close", f"SMA{win}", "EMA20"],
    title=f"{ticker} Price"
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df.tail(60))

# ---------------- Trade Form ----------------
import datetime as dt
st.subheader("เพิ่มธุรกรรม")
with st.form("trade_form", clear_on_submit=True):
    d   = st.date_input("วันที่", dt.date.today())
    sym = st.text_input("สัญลักษณ์", ticker).upper().strip()
    side= st.selectbox("ประเภท", ["BUY", "SELL"])
    qty = st.number_input("จำนวน", min_value=1, value=10)
    price_input = st.number_input("ราคา/หน่วย", min_value=0.0, value=float(df["Close"].iloc[-1]))
    fee = st.number_input("ค่าธรรมเนียม", min_value=0.0, value=0.0)
    submitted = st.form_submit_button("บันทึกธุรกรรม")
    if submitted:
        msg = safe_add_trade(pf_id, sym, d.strftime("%Y-%m-%d"),
                         side, int(qty), float(price_input), float(fee))
        if msg.startswith("✅"):
            st.success(msg)
        else:
            st.error(msg)


# ---------------- Latest trades ----------------
st.subheader("ธุรกรรมล่าสุด")
trades = read_trades_df(pf_id)
st.dataframe(trades)
