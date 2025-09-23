# pages/2_Portfolio_Dashboard.py  — หน้า 2: Portfolio Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
from db_utils import (
    init_db, ensure_portfolio, read_symbols_for_portfolio, read_trades_df,
    average_cost_position, load_prices_df, compute_equity_curve,
    compute_nav_now, compute_growth_vs_initial
)

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("Portfolio Dashboard")

# --------- Sidebar ---------
portfolio_key = st.sidebar.text_input("Portfolio Name", "portfolio1")
init_db()
pf_id = ensure_portfolio(portfolio_key)

# --------- Hero metrics ---------
snap = compute_nav_now(pf_id)
growth = compute_growth_vs_initial(pf_id)

c1, c2, c3, c4, c5 = st.columns(5)
if growth["growth_pct"] is None:
    c1.metric("โตขึ้น (เทียบก้อนแรก)", "—")
else:
    c1.metric("โตขึ้น (เทียบก้อนแรก)", f"{growth['growth_pct']:.2f}%")
c2.metric("NAV ปัจจุบัน", f"{snap['nav_now']:,.2f}")
c3.metric("เงินสด (Cash)", f"{snap['cash_now']:,.2f}")
c4.metric("Realized P&L", f"{snap['realized_pnl']:,.2f}")
c5.metric("Unrealized P&L", f"{snap['unrealized_pnl']:,.2f}")

st.caption(
    "คำอธิบาย: โตขึ้น% คำนวณจาก NAV ปัจจุบันเทียบเงินฝากครั้งแรก (Initial Deposit). "
    "สำหรับกรณีมีเงินฝาก/ถอนเพิ่มเติมภายหลัง ค่าดังกล่าวไม่ได้ปรับแบบ TWR/MWR (เวอร์ชันแรก)"
)

# --------- Equity Curve ---------
st.subheader("📈 Equity Curve (ตั้งแต่เงินฝากครั้งแรก)")
curve = compute_equity_curve(pf_id)
if curve is None or curve.empty:
    st.info("ยังไม่มีข้อมูลพอสำหรับคำนวณเส้นเติบโต (ฝากเงิน/บันทึกธุรกรรม และบันทึกราคาในหน้า 1)")
else:
    curve_reset = curve.reset_index().rename(columns={"index": "Date"}) if "index" in curve.reset_index().columns else curve.reset_index()
    fig_nav = px.line(curve_reset, x=curve_reset.columns[0], y=["NAV", "Cash", "MV"], title="NAV / Cash / MV")
    st.plotly_chart(fig_nav, use_container_width=True)
    st.dataframe(curve.tail(10))

# --------- Allocation Pie + Holdings ---------
st.subheader("🧩 สัดส่วนพอร์ต (Allocation) และรายการถือครอง (Holdings)")

symbols = read_symbols_for_portfolio(pf_id)
alloc_rows = []
hold_rows = []

for sym in symbols:
    tdf = read_trades_df(pf_id, sym)
    qty, avg_cost, realized = average_cost_position(tdf)
    if qty == 0 and tdf.empty:
        continue
    px_df = load_prices_df(sym)
    last = float(px_df["Close"].iloc[-1]) if not px_df.empty else avg_cost
    mv = qty * last
    unrl = (last - avg_cost) * qty
    if qty != 0:
        alloc_rows.append({"Symbol": sym, "MarketValue": mv})
    hold_rows.append({
        "Symbol": sym,
        "Qty": qty,
        "AvgCost": avg_cost,
        "Last": last,
        "MarketValue": mv,
        "UnrealizedPnL": unrl,
        "RealizedPnL": realized
    })

alloc_df = pd.DataFrame(alloc_rows)
hold_df = pd.DataFrame(hold_rows)

col1, col2 = st.columns([1, 2])
with col1:
    if alloc_df.empty or alloc_df["MarketValue"].sum() == 0:
        st.info("ยังไม่มีสถานะคงเหลือสำหรับวาด Pie")
    else:
        fig_pie = px.pie(alloc_df, names="Symbol", values="MarketValue", title="Portfolio Allocation (MV)")
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.dataframe(hold_df.sort_values("MarketValue", ascending=False), use_container_width=True)
