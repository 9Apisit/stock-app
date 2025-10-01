# pages/3_Analysis_ML_Insights.py
# -----------------------------------------------------------
# หน้า 3: การวิเคราะห์เชิงลึก + ML (UI Refined)
# - Tab 1: Stock Prediction (Regression)
# - Tab 2: Stock Recommendation (Correlation-based)
# - Tab 3: Ensemble / Bagging / Boosting (Forecasting Result)
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
    load_prices_df, upsert_prices,
)

# ML (Regression / Ensemble)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# สำหรับแท็บ Ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor

import yfinance as yf
from typing import Optional

# ---------------- Page config ----------------
st.set_page_config(page_title="Analysis & ML Insights", layout="wide")
st.title("📊 Analysis & Machine Learning Insights")

# ---------------- Sidebar: เลือกพอร์ต/สัญลักษณ์ ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    portfolio_key = st.text_input("Portfolio Key", "portfolio1", help="ตั้งชื่อ/เลือกพอร์ตสำหรับดึง symbol จากฐานข้อมูล")
    init_db()
    pf_id = ensure_portfolio(portfolio_key)

    st.divider()
    st.caption("เลือกช่วงเวลา")
    default_start = date.today() - timedelta(days=365)
    start = st.date_input("Start", default_start, help="")
    end = st.date_input("End", date.today(), help="")

# ---------------- Helper: ดึงราคาแบบยืดหยุ่น ----------------
@st.cache_data(show_spinner=False)
def fetch_yf(tk: str, start_d: date, end_d: date):
    """ดึง OHLCV จาก yfinance (auto_adjust=True)"""
    df = yf.download(tk, start=start_d, end=end_d, auto_adjust=True, progress=False)
    return df if df is not None else pd.DataFrame()


def get_close_series(symbol: str, start_d: Optional[date] = None, end_d: Optional[date] = None) -> pd.Series:
    """ดึงราคาจาก DB หรือ yfinance คืนค่าเป็น Series ของ Close"""
    df = load_prices_df(symbol)
    if df is not None and not df.empty:
        s = df.copy()
        if "Date" in s.columns:
            s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
            s = s.dropna(subset=["Date"]).sort_values("Date")
            if start_d:
                s = s[s["Date"] >= pd.to_datetime(start_d)]
            if end_d:
                s = s[s["Date"] <= pd.to_datetime(end_d)]
            s = s.set_index("Date")
            if "Close" in s.columns:
                return pd.to_numeric(s["Close"], errors="coerce").dropna().astype(float)

    # ถ้าใน DB ไม่มี → ดึงจาก yfinance แล้ว upsert เฉพาะ Close
    yf_df = fetch_yf(symbol, start, end)
    if yf_df is not None and not yf_df.empty:
        y2 = yf_df.reset_index()
        date_col = "Date" if "Date" in y2.columns else ("Datetime" if "Datetime" in y2.columns else None)
        if date_col and "Close" in y2.columns:
            y2 = y2[[date_col, "Close"]].rename(columns={date_col: "Date"})
            y2["Date"] = pd.to_datetime(y2["Date"], errors="coerce")
            y2 = y2.dropna(subset=["Date"])  # เอาเฉพาะวันที่ valid
            y2["Close"] = pd.to_numeric(y2["Close"], errors="coerce")
            y2 = y2.dropna(subset=["Close"])  # เอาเฉพาะราคาที่เป็นตัวเลข
            upsert_prices(symbol, y2[["Date", "Close"]])  # เก็บ cache ลง DB (เฉพาะ Close)
            return y2.set_index("Date")["Close"].astype(float)

    return pd.Series(dtype="float64")

# ---------------- Tabs ----------------
tab_pred, tab_reco, tab_ens = st.tabs([
    "🔮 Prediction (Regression)",
    "🧭 Recommendation (Correlation)",
    "🧪 Ensemble (Bagging & Boosting)",
])

# =============================================================================
# TAB 1: PREDICTION (REGRESSION)
# =============================================================================
with tab_pred:
    st.subheader("🔮 Stock Price Prediction — Linear Regression")
    st.caption("โมเดลเชิงเส้นพื้นฐานเพื่อพยากรณ์ราคาปิดวันถัดไป (t+1)")

    symbols_in_pf = read_symbols_for_portfolio(pf_id)
    default_sym = symbols_in_pf[0] if symbols_in_pf else "AAPL"
    target_symbol = st.selectbox(
        "เลือกสัญลักษณ์ที่จะพยากรณ์",
        options=[default_sym] + ["AAPL", "MSFT", "NVDA", "PTT.BK", "AOT.BK"],
        index=0,
    )

    s_close = get_close_series(target_symbol, start, end)

    if s_close.empty or len(s_close) < 50:
        st.warning("ข้อมูลราคาน้อยเกินไปสำหรับการเทรนโมเดล (ต้องการอย่างน้อย ~50 จุด). ลองเลือกสัญลักษณ์อื่นหรือขยายช่วงเวลา")
    else:
        df = s_close.rename("Close").to_frame().copy()
        df.index.name = "Date"
        df["Return"] = df["Close"].pct_change()
        for lag in [1, 2, 3, 5]:
            df[f"lag_{lag}"] = df["Close"].shift(lag)
        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_10"] = df["Close"].rolling(10).mean()
        df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["target_next"] = df["Close"].shift(-1)
        df_model = df.dropna().copy()

        feature_cols = [c for c in df_model.columns if c != "target_next"]
        X = df_model[feature_cols].values
        y = df_model["target_next"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        df_plot = df_model.iloc[-len(y_test):].copy()
        df_plot["Predicted"] = y_pred
        df_plot["Actual"] = y_test

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:,.4f}", help="Root Mean Squared Error — ค่าน้อยกว่า ดีกว่า")
        latest_row = df.dropna().iloc[-1:]
        next_pred = float(model.predict(latest_row[feature_cols].values)[0])
        c2.metric("Predicted Next Close", f"{next_pred:,.2f}")
        c3.metric("Sample Size", f"{len(df_model):,}")

        fig_pred = px.line(
            df_plot.reset_index(), x="Date", y=["Actual", "Predicted"],
            title=f"{target_symbol} — Actual vs Predicted (Hold-out)", labels={"value": "Price"}
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        with st.expander("ดูตารางฟีเจอร์ (ท้าย 10 แถว)"):
            st.dataframe(df.tail(10))

# =============================================================================
# TAB 2: RECOMMENDATION (CORRELATION-BASED)
# =============================================================================
with tab_reco:
    st.subheader("🧭 Stock Recommendation — Correlation Similarity")
    st.caption("แนะนำหุ้นที่เคลื่อนไหวคล้ายกับหุ้นอ้างอิง โดยวัดจากสหสัมพันธ์ของผลตอบแทนรายวัน")

    st.caption("ป้อนรายการหุ้น เช่น AAPL,MSFT,NVDA,GOOGL,AMZN หรือ .BK สำหรับหุ้นไทย")
    universe_raw = st.text_input(
        "Universe (Comma-separated)",
        value="AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,PTT.BK,AOT.BK,SCB.BK,KBANK.BK",
    )
    universe = [x.strip().upper() for x in universe_raw.split(",") if x.strip()]

    held = read_symbols_for_portfolio(pf_id)
    st.write("Symbols in portfolio:", ", ".join(held) if held else "— (ยังไม่มีธุรกรรม)")

    closes = {}
    for sym in universe:
        s = get_close_series(sym, start, end)
        if not s.empty:
            closes[sym] = s

    if len(closes) < 3:
        st.warning("ยังดึงข้อมูลราคาใน universe ได้ไม่พอ ลองเพิ่มรายการหุ้นหรือลอง fetch ใหม่")
    else:
        price_df = pd.DataFrame(closes).sort_index().dropna(how="all")
        ret_df = price_df.pct_change().dropna(how="all")
        ref = [h for h in held if h in ret_df.columns] if held else [x for x in list(ret_df.columns)[:1]]
        st.write("Reference symbols (for similarity):", ", ".join(ref))

        if ref:
            corr_matrix = ret_df.corr()
            corr_scores = corr_matrix[ref].mean(axis=1).drop(labels=ref, errors="ignore").sort_values(ascending=False)
            N = st.slider("Top-N Recommendations", 3, 10, 5)
            candidates = corr_scores.index.tolist()
            recommend = [c for c in candidates if c not in held][:N]

            st.subheader("ผลลัพธ์การแนะนำ (Similarity by Correlation)")
            res = pd.DataFrame({"Symbol": recommend, "Similarity": corr_scores.loc[recommend].values})
            st.dataframe(res, use_container_width=True)

            if len(recommend) > 0:
                top1 = recommend[0]
                plot_syms = ref + [top1]
                norm = (price_df[plot_syms].dropna() / price_df[plot_syms].dropna().iloc[0])
                fig_cmp = px.line(norm.reset_index(), x="Date", y=plot_syms,
                                  title=f"Normalized Price — {', '.join(plot_syms)}")
                st.plotly_chart(fig_cmp, use_container_width=True)

        with st.expander("ดูตารางราคาที่ใช้คำนวณ (tail 10)"):
            st.dataframe(price_df.tail(10))

# =============================================================================
# TAB 3: ENSEMBLE / BAGGING & BOOSTING (FORECASTING RESULT)
# =============================================================================
with tab_ens:
    st.caption("ENSEMBLE / BAGGING & BOOSTING")
    st.subheader("🧪 Forecasting Result")
    st.caption("เปรียบเทียบโมเดลหลายตระกูล และคำนวณค่าเฉลี่ยแบบง่ายเพื่อใช้เป็น Ensemble")

    ens_symbol = st.text_input("Ticker สำหรับ Ensemble", value="AAPL").strip().upper()
    lag_n = st.slider("จำนวน Lags (วัน)", 3, 30, 10, help="จำนวนวันย้อนหลังที่ใช้สร้างฟีเจอร์ lag ของ Close")
    roll_n = st.slider("Rolling Window", 3, 60, 10, help="หน้าต่างสำหรับคำนวณค่าเฉลี่ย/ความผันผวน")
    test_ratio = st.slider("Test Size (%)", 10, 40, 20, help="สัดส่วนชุดทดสอบจากข้อมูลทั้งหมด")

    # ---------- ดึง OHLCV ที่ทนทาน ----------
    @st.cache_data(show_spinner=False)
    def fetch_ohlc_clean(tk, start_d, end_d):
        def _normalize_cols(df0: pd.DataFrame) -> pd.DataFrame:
            df = df0.copy()
            if df.empty:
                return df
            new_cols = {c: str(c).strip().lower().replace(" ", "") for c in df.columns}
            df.rename(columns=new_cols, inplace=True)
            colmap = {}
            if "open" in df.columns: colmap["open"] = "Open"
            if "high" in df.columns: colmap["high"] = "High"
            if "low" in df.columns: colmap["low"] = "Low"
            if "close" in df.columns: colmap["close"] = "Close"
            if "adjclose" in df.columns: colmap["adjclose"] = "Adj Close"
            if "volume" in df.columns: colmap["volume"] = "Volume"
            if "highprice" in df.columns: colmap["highprice"] = "High"
            if "lowprice" in df.columns: colmap["lowprice"] = "Low"
            if "closingprice" in df.columns: colmap["closingprice"] = "Close"
            if "turnover" in df.columns and "volume" not in df.columns: colmap["turnover"] = "Volume"
            df.rename(columns=colmap, inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df.set_index("Date", inplace=True)
            elif "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
                df.set_index("Datetime", inplace=True)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            if len(df.index) == 0:
                return pd.DataFrame()
            df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.ffill()
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
            return df

        df = yf.download(tk, start=start_d, end=end_d, auto_adjust=True, progress=False)
        if df is None or df.empty:
            try:
                df = yf.Ticker(tk).history(start=start_d, end=end_d, auto_adjust=True)
            except Exception:
                df = pd.DataFrame()
        df = _normalize_cols(df)
        if df is None or df.empty:
            return pd.DataFrame()
        try:
            df = df.asfreq("B")
        except Exception:
            pass

        def _as_series(x: pd.Series | pd.DataFrame) -> pd.Series:
            return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
        def _has(col: str) -> bool:
            if col not in df.columns:
                return False
            vals = pd.to_numeric(_as_series(df[col]), errors="coerce")
            return bool(vals.notna().any())

        have_close, have_open, have_high, have_low = _has("Close"), _has("Open"), _has("High"), _has("Low")
        if (not have_open) and have_close:
            df["Open"] = _as_series(df["Close"]); have_open = True
        if (not have_high) and (have_open and have_close):
            df["High"] = pd.concat([_as_series(df["Open"]), _as_series(df["Close"])], axis=1).max(axis=1); have_high = True
        if (not have_low) and (have_open and have_close):
            df["Low"] = pd.concat([_as_series(df["Open"]), _as_series(df["Close"])], axis=1).min(axis=1); have_low = True
        if (not have_high) and have_close:
            df["High"] = _as_series(df["Close"]) ; have_high = True
        if (not have_low) and have_close:
            df["Low"] = _as_series(df["Close"])  ; have_low  = True

        need = [c for c in ["High", "Low", "Close", "Open", "Volume"] if c in df.columns]
        df = df[need].dropna(how="all")
        return df

    ohlc = fetch_ohlc_clean(ens_symbol, start, end)
    if ohlc.empty or "Close" not in ohlc.columns:
        st.error("ดึงข้อมูลราคาไม่สำเร็จจาก yfinance (ไม่มี Close เลย) — ลองเปลี่ยนสัญลักษณ์หรือลองช่วงเวลาอื่น")
        st.stop()

    st.caption(f"ข้อมูลอินพุตสำหรับ Ensemble: {ens_symbol}")
    with st.expander("🔎 Preview OHLC (tail 10)"):
        st.dataframe(ohlc.tail(10))
        st.write("Columns:", list(ohlc.columns))

    # -------- Feature Engineering สำหรับ High / Low / Close --------
    def make_features(frame: pd.DataFrame, target_col: str, lag_n: int, roll_n: int) -> pd.DataFrame:
        def _col_as_series(df: pd.DataFrame, col: str) -> pd.Series:
            col_obj = df[col]
            if isinstance(col_obj, pd.DataFrame):
                col_obj = col_obj.iloc[:, 0]
            return pd.to_numeric(col_obj, errors="coerce")
        s_close = _col_as_series(frame, "Close")
        s_target = _col_as_series(frame, target_col)
        X = pd.DataFrame(index=frame.index)
        X["close"] = s_close
        X["ret1"] = s_close.pct_change()
        for i in range(1, lag_n + 1):
            X[f"close_lag{i}"] = s_close.shift(i)
        X[f"sma{roll_n}"] = s_close.rolling(roll_n).mean()
        X[f"ema{roll_n}"] = s_close.ewm(span=roll_n, adjust=False).mean()
        X[f"vol{roll_n}"] = s_close.pct_change().rolling(roll_n).std()
        for i in range(1, min(5, lag_n) + 1):
            X[f"{target_col.lower()}_lag{i}"] = s_target.shift(i)
        y = s_target.shift(-1).copy(); y.name = f"{target_col}_t+1"
        data = pd.concat([X, y], axis=1).dropna()
        return data

    targets = ["High", "Low", "Close"]
    data_map = {t: make_features(ohlc, t, lag_n, roll_n) for t in targets}

    def time_split(data: pd.DataFrame, test_ratio: float):
        n = len(data); n_test = int(np.ceil(n * (test_ratio / 100.0))); n_train = max(1, n - n_test)
        train = data.iloc[:n_train].copy(); test = data.iloc[n_train:].copy()
        X_train, y_train = train.drop(columns=[train.columns[-1]]), train.iloc[:, -1]
        X_test, y_test = test.drop(columns=[test.columns[-1]]), test.iloc[:, -1]
        return X_train, X_test, y_train, y_test

    # ---------- โมเดล & ป้ายชื่อแบบมืออาชีพ ----------
    MODEL_LABELS = {
        "nn": "Neural Network (MLP-64×32)",
        "svm": "Support Vector Machine (SVM)",
        "ms": "Gradient Boosting (GBR)",
        "mlp": "Neural Network (MLP-128)",
        "bagging": "Bagging Regressor",
        "rf": "Random Forest (RF)",
        "gbr": "Gradient Boosting (GBR)",
    }

    def build_models(random_state=42):
        return {
            "nn": Pipeline([("scaler", StandardScaler()), ("est", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state))]),
            "svm": Pipeline([("scaler", StandardScaler()), ("est", SVR(C=5.0, epsilon=0.1))]),
            "ms": GradientBoostingRegressor(random_state=random_state),
            "mlp": Pipeline([("scaler", StandardScaler()), ("est", MLPRegressor(hidden_layer_sizes=(128,), max_iter=600, random_state=random_state))]),
            "bagging": BaggingRegressor(n_estimators=10, random_state=random_state),
            "rf": RandomForestRegressor(n_estimators=300, random_state=random_state),
            "gbr": GradientBoostingRegressor(random_state=random_state),
        }

    models = build_models()

    def train_and_predict_for_target(data: pd.DataFrame, models: dict, test_ratio: float):
        X_train, X_test, y_train, y_test = time_split(data, test_ratio)
        preds_test, rmses, future_preds = {}, {}, {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmses[name] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            X_future = data.drop(columns=[data.columns[-1]]).iloc[[-1]]
            future_preds[name] = float(model.predict(X_future)[0])
        return {"rmse": rmses, "future": future_preds, "last_test_true": float(y_test.iloc[-1]) if len(y_test) else np.nan}

    results = {t: train_and_predict_for_target(data_map[t], models, test_ratio) for t in targets}

    # ---------- ตารางผลลัพธ์แบบอ่านง่าย ----------
    focus_keys = ["nn", "svm", "ms", "mlp"]
    method_table = pd.DataFrame(index=["high", "low", "close"], columns=[MODEL_LABELS[k] for k in focus_keys], dtype=float)
    for t in targets:
        row = t.lower()
        for k in focus_keys:
            method_table.loc[row, MODEL_LABELS[k]] = results[t]["future"].get(k, np.nan)

    expected = method_table.mean(axis=1).rename("Expected")

    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🧩 Ensemble (Simple Average)")
    c1.metric("Expected — Close", f"{expected.loc['close']:.6f}")
    c2.metric("Expected — High", f"{expected.loc['high']:.6f}")
    c3.metric("Expected — Low", f"{expected.loc['low']:.6f}")

    st.subheader("Model Outputs (Focus Methods)")
    st.dataframe(method_table.style.format("{:.6f}"), use_container_width=True)

    st.subheader("Expected (Simple Average of Focus Methods)")
    expected_df = pd.DataFrame({"High": [expected.loc["high"]], "Low": [expected.loc["low"]], "Close": [expected.loc["close"]]})
    st.dataframe(expected_df.style.format("{:.6f}"), use_container_width=True)

    # RMSE Summary
    rmse_rows = []
    for t in targets:
        row = {MODEL_LABELS.get(k, k): v for k, v in results[t]["rmse"].items()}
        row.update({"Target": t})
        rmse_rows.append(row)
    rmse_df = pd.DataFrame(rmse_rows).set_index("Target")
    st.subheader("RMSE by Model and Target (lower is better)")
    st.dataframe(rmse_df.style.format("{:.4f}"), use_container_width=True)

    # Future prediction (all models)
    future_rows = []
    for t in targets:
        row = {MODEL_LABELS.get(k, k): v for k, v in results[t]["future"].items()}
        row.update({"Target": t})
        future_rows.append(row)
    future_df = pd.DataFrame(future_rows).set_index("Target")
    st.subheader("Next-day Forecast by Model (All Families)")
    st.dataframe(future_df.style.format("{:.6f}"), use_container_width=True)

#     st.info(
#         """หมายเหตุ:
# - Focus Methods คือชุดโมเดลที่ใช้สำหรับคำนวณ Expected (ค่าเฉลี่ยอย่างง่าย)
# - RMSE = Root Mean Squared Error; ยิ่งต่ำยิ่งดี
# - ค่าทำนายเป็นการสาธิตเชิงการศึกษา ไม่ใช่คำแนะนำการลงทุน
# """
#     )
