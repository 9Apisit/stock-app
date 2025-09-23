# pages/3_Analysis_ML_Insights.py
# -----------------------------------------------------------
# หน้า 3: การวิเคราะห์เชิงลึก + ML
# - Tab 1: Stock Prediction (Regression)
# - Tab 2: Stock Recommendation (Correlation-based)
# ใช้ข้อมูลราคาจากตาราง prices ใน SQLite (ผ่าน db_utils.load_prices_df)
# ถ้าไม่มีใน DB จะมี option ให้ดึงจาก yfinance (เพื่อ demo)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

from db_utils import (
    init_db, ensure_portfolio, read_symbols_for_portfolio,
    load_prices_df, upsert_prices
)

# ML (Regression)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import yfinance as yf

# ---------------- Page config ----------------
st.set_page_config(page_title="Analysis & ML Insights", layout="wide")
st.title("Analysis & ML Insights")

# ---------------- Sidebar: เลือกพอร์ต/สัญลักษณ์ ----------------
portfolio_key = st.sidebar.text_input("Portfolio Key", "portfolio1")
init_db()
pf_id = ensure_portfolio(portfolio_key)

st.sidebar.markdown("—")
st.sidebar.caption("กรณีต้องดึงราคาจาก yfinance")
default_start = date.today() - timedelta(days=365)
start = st.sidebar.date_input("Start (fetch)", default_start)
end = st.sidebar.date_input("End (fetch)", date.today())

# ---------------- Helper: ดึงราคาแบบยืดหยุ่น ----------------
@st.cache_data(show_spinner=False)
def fetch_yf(tk: str, start_d: date, end_d: date):
    df = yf.download(tk, start=start_d, end=end_d, auto_adjust=True, progress=False)
    return df if df is not None else pd.DataFrame()

def get_close_series(symbol: str, start_d: date | None = None, end_d: date | None = None) -> pd.Series:
    """ดึงราคาจาก DB หรือ yfinance คืนค่าเป็น Series"""
    df = load_prices_df(symbol)
    if df is not None and not df.empty:
        s = df.copy()
        if start_d:
            s = s[s["Date"] >= pd.to_datetime(start_d)]
        if end_d:
            s = s[s["Date"] <= pd.to_datetime(end_d)]
        # ✅ force เป็น Series
        s = s.sort_values("Date").set_index("Date")["Close"].astype(float)
        return s

    # ถ้าใน DB ไม่มี → ดึงจาก yfinance (เพื่อ demo) แล้ว upsert
    yf_df = fetch_yf(symbol, start, end)
    if yf_df is not None and not yf_df.empty:
        # ปกติ yfinance ให้คอลัมน์ 'Close'
        y2 = yf_df.reset_index()[["Date", "Close"]].rename(columns={"Date": "Date", "Close": "Close"})
        upsert_prices(symbol, y2)  # เก็บ cache ลง DB
        return y2.set_index("Date")["Close"].astype(float)

    return pd.Series(dtype="float64")

# ---------------- Tabs ----------------
tab_pred, tab_reco = st.tabs(["🔮 Prediction", "🧭 Recommendation"])

# =============================================================================
# TAB 1: PREDICTION (REGRESSION)
# =============================================================================
with tab_pred:
    st.subheader("🔮 Stock Price Prediction (Linear Regression)")

    # --- เลือกสัญลักษณ์เพื่อพยากรณ์ ---
    symbols_in_pf = read_symbols_for_portfolio(pf_id)
    default_sym = symbols_in_pf[0] if symbols_in_pf else "AAPL"
    target_symbol = st.selectbox("เลือกสัญลักษณ์ที่จะพยากรณ์", options=[default_sym] + ["AAPL","MSFT","NVDA","PTT.BK","AOT.BK"], index=0)

    # --- ดึงข้อมูลราคาปิด ---
    s_close = get_close_series(target_symbol, start, end)

    if s_close.empty or len(s_close) < 50:
        st.warning("ข้อมูลราคาน้อยเกินไปสำหรับการเทรนโมเดล (ต้องการอย่างน้อย ~50 จุด). ลองเลือกสัญลักษณ์อื่นหรือขยายช่วงเวลา")
    else:
        # --- สร้างฟีเจอร์สำหรับ Regression แบบง่าย ---
        df = s_close.rename("Close").to_frame().copy()
        df.index.name = "Date"
        df["Return"] = df["Close"].pct_change()
        # ฟีเจอร์: lag ราคาย้อนหลัง + moving averages
        for lag in [1, 2, 3, 5]:
            df[f"lag_{lag}"] = df["Close"].shift(lag)
        df["sma_5"]  = df["Close"].rolling(5).mean()
        df["sma_10"] = df["Close"].rolling(10).mean()
        df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()

        # เป้าหมาย: พยากรณ์ราคาถัดไป 1 วัน (t+1)
        df["target_next"] = df["Close"].shift(-1)

        # ทำความสะอาด null
        df_model = df.dropna().copy()

        # --- แบ่ง train/test ---
        feature_cols = [c for c in df_model.columns if c not in ["target_next"]]
        X = df_model[feature_cols].values
        y = df_model["target_next"].values

        # ขนาดชุดทดสอบ 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # --- เทรน Linear Regression ---
        model = LinearRegression()
        model.fit(X_train, y_train)

        # --- ประเมิน ---
        y_pred = model.predict(X_test)
        rmse = (mean_squared_error(y_test, y_pred)) ** 0.5

        # --- สร้างเส้นคาดการณ์ช่วงท้าย (ลากขึ้นกราฟ) ---
        df_plot = df_model.iloc[-len(y_test):].copy()
        df_plot["Predicted"] = y_pred
        df_plot["Actual"] = y_test

        c1, c2 = st.columns(2)
        c1.metric("RMSE (lower is better)", f"{rmse:,.4f}")
        c2.caption("หมายเหตุ: โมเดลนี้เป็น baseline regression ง่าย ๆ เพื่อสาธิตแนวคิดเท่านั้น")

        # --- กราฟราคาจริงเทียบค่าพยากรณ์ ---
        fig_pred = px.line(
            df_plot.reset_index(), x="Date",
            y=["Actual", "Predicted"],
            title=f"{target_symbol} — Actual vs Predicted (hold-out)",
            labels={"value": "Price"}
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # --- คาดการณ์วันถัดไปจากข้อมูลล่าสุด (inference) ---
        latest_row = df.dropna().iloc[-1:]
        X_last = latest_row[feature_cols].values
        next_pred = float(model.predict(X_last)[0])
        st.success(f"คาดการณ์ราคาวันถัดไปของ {target_symbol}: **{next_pred:,.2f}**")

        # --- แสดงฟีเจอร์/ตารางเล็กน้อย ---
        with st.expander("ดูตารางฟีเจอร์ (ท้าย 10 แถว)"):
            st.dataframe(df.tail(10))

# =============================================================================
# TAB 2: RECOMMENDATION (CORRELATION-BASED)
# =============================================================================
with tab_reco:
    st.subheader("🧭 Stock Recommendation (Correlation-based)")

    # --- นิยาม universe ของหุ้นให้เลือก (ผู้ใช้แก้ไขได้) ---
    st.caption("ใส่รายการหุ้นเป็นคอมมา เช่น AAPL,MSFT,NVDA,GOOGL,AMZN หรือ .BK สำหรับหุ้นไทย")
    universe_raw = st.text_input(
        "Universe (Comma-separated)",
        value="AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,PTT.BK,AOT.BK,SCB.BK,KBANK.BK"
    )
    universe = [x.strip().upper() for x in universe_raw.split(",") if x.strip()]

    # --- หุ้นที่ถืออยู่ในพอร์ต (อาจว่างก็ได้) ---
    held = read_symbols_for_portfolio(pf_id)
    st.write("Symbols in portfolio:", ", ".join(held) if held else "— (ยังไม่มีธุรกรรม)")

    # --- ดึง series ราคาปิดของหุ้นใน universe ---
    closes = {}
    for sym in universe:
        s = get_close_series(sym, start, end)
        if s.empty:
            continue
        closes[sym] = s

    if len(closes) < 3:
        st.warning("ยังดึงข้อมูลราคาใน universe ได้ไม่พอ ลองเพิ่มรายการหุ้นหรือลอง fetch ใหม่")
    else:
        # --- รวมเป็น DataFrame เดียว (align ตามวันที่) ---
        price_df = pd.DataFrame(closes).sort_index()
        # กรองวันที่ให้แน่น (drop แถวที่ว่างทั้งบรรทัด)
        price_df = price_df.dropna(how="all")

        # --- คำนวณผลตอบแทนรายวันเพื่อนำไปหา correlation ---
        ret_df = price_df.pct_change().dropna(how="all")

        # --- หา correlation เฉลี่ยกับหุ้นที่ถืออยู่ (หรือเลือกสัญลักษณ์ตั้งต้น) ---
        if held:
            ref = [h for h in held if h in ret_df.columns]
        else:
            # ถ้ายังไม่มี holdings เลือกสัญลักษณ์ตั้งต้นจาก universe ตัวแรก ๆ
            ref = [x for x in list(ret_df.columns)[:1]]

        st.write("Reference symbols (for similarity):", ", ".join(ref))

        if not ref:
            st.info("ยังไม่มี reference ที่ซ้อนทับในราคาที่ดึงมา")
        else:
            # คำนวณ correlation เฉลี่ยของแต่ละ candidate เทียบกับชุด ref
            corr_matrix = ret_df.corr()
            corr_scores = corr_matrix[ref].mean(axis=1).drop(labels=ref, errors="ignore")
            corr_scores = corr_scores.sort_values(ascending=False)

            # แนะนำ Top-N ที่ยังไม่ได้ถือ
            N = st.slider("Top-N Recommendations", min_value=3, max_value=10, value=5)
            candidates = corr_scores.index.tolist()
            # ข้ามตัวที่ถืออยู่แล้ว
            recommend = [c for c in candidates if c not in held][:N]

            st.subheader("ผลลัพธ์การแนะนำ (Similarity by Correlation)")
            res = pd.DataFrame({
                "Symbol": recommend,
                "Similarity": corr_scores.loc[recommend].values
            })
            st.dataframe(res, use_container_width=True)

            # --- กราฟเปรียบเทียบ normalized price ของ ref vs top-1 recommendation ---
            if len(recommend) > 0:
                top1 = recommend[0]
                plot_syms = ref + [top1]
                sub = price_df[plot_syms].dropna()
                # normalize ให้เริ่มที่ 1.0 เพื่อเทียบรูปทรง
                norm = sub / sub.iloc[0]
                fig_cmp = px.line(norm.reset_index(), x="Date", y=plot_syms,
                                  title=f"Normalized Price — {', '.join(plot_syms)}")
                st.plotly_chart(fig_cmp, use_container_width=True)

        with st.expander("ดูตารางราคาที่ใช้คำนวณ (tail 10)"):
            st.dataframe(price_df.tail(10))
