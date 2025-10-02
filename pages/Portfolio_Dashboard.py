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
# # ========= HERO PREVIEW (no inputs, no DB) =========
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# np.random.seed(7)  # ทำให้ภาพคงที่ทุกครั้ง

# # 1) กำหนดพารามิเตอร์สมมติ (ปรับให้ภาพสวย/ดูดี)
# init_cap = 250_000                 # เงินตั้งต้น
# equity_ratio = 72                  # สัดส่วนถือหุ้น (%)
# exp_annual_ret = 14                # คาดหวังผลตอบแทนต่อปี (%)
# annual_vol = 18                    # ความผันผวนต่อปี (%)
# start = pd.to_datetime("2025-01-01")
# end   = pd.to_datetime("2025-06-30")

# # 2) สร้างวันทำการ + ผลตอบแทนรายวัน
# bdays = pd.bdate_range(start, end)
# n = len(bdays)
# mu_d = (exp_annual_ret / 100) / 252.0
# sigma_d = (annual_vol / 100) / np.sqrt(252.0)
# daily_ret = np.random.normal(mu_d, sigma_d, n)

# # 3) สร้าง “เงินฝาก/ถอน” เล็กน้อยให้ภาพดูมีชีวิต (ไม่กระทบข้อมูลจริง)
# flows = np.zeros(n)
# if n > 40:
#     # เติมสัก 4 จุดแบบสุ่ม ±0.5–2.5% ของเงินตั้งต้น
#     for i in np.linspace(10, n-10, 4, dtype=int):
#         flows[i] = np.random.choice([1, -1]) * np.random.uniform(0.005, 0.025) * init_cap
# cumulative_flows = np.cumsum(flows)

# # 4) คำนวณ NAV / Cash / MV + Growth(%)
# growth_factor = np.cumprod(1.0 + daily_ret)
# NAV = init_cap * growth_factor + cumulative_flows
# MV  = NAV * (equity_ratio / 100.0)
# Cash = NAV - MV
# initial_capital = float(NAV[0])
# GrowthPct = (NAV / initial_capital - 1.0) * 100.0

# df_show = pd.DataFrame(
#     {"NAV": NAV, "Cash": Cash, "MV": MV, "GrowthPct": GrowthPct},
#     index=bdays
# )

# # 5) พล็อตกราฟสวย ๆ แกนคู่
# fig_hero = make_subplots(specs=[[{"secondary_y": True}]])
# for c in ["NAV", "Cash", "MV"]:
#     fig_hero.add_trace(
#         go.Scatter(x=df_show.index, y=df_show[c], mode="lines", name=c),
#         secondary_y=False
#     )
# fig_hero.add_trace(
#     go.Scatter(
#         x=df_show.index, y=df_show["GrowthPct"], mode="lines",
#         name="Growth (%)", line=dict(dash="dash")
#     ),
#     secondary_y=True
# )

# fig_hero.update_layout(
#     title="Sample Portfolio • NAV / Cash / MV + Growth (%)",
#     template="plotly_white",
#     hovermode="x unified",
#     margin=dict(l=40, r=40, t=60, b=40),
#     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
# )
# fig_hero.update_xaxes(title_text="Date")
# fig_hero.update_yaxes(title_text="Value (฿)", secondary_y=False)
# fig_hero.update_yaxes(title_text="Growth (%)", secondary_y=True)

# st.plotly_chart(fig_hero, use_container_width=True)

# with st.expander("ตัวอย่างข้อมูลที่ใช้แสดงผล (ท้าย 10 แถว)"):
#     st.dataframe(df_show.tail(10))
# # ========= END HERO PREVIEW =========


# --------- Equity Curve ---------
st.subheader("📈 Equity Curve (ตั้งแต่เงินฝากครั้งแรก)")
curve = compute_equity_curve(pf_id)

if curve is None or curve.empty:
    st.info("ยังไม่มีข้อมูลพอสำหรับคำนวณเส้นเติบโต (ฝากเงิน/บันทึกธุรกรรม และบันทึกราคาในหน้า 1)")
else:
    # ทำ index ให้เป็น datetime และ reset ให้มีคอลัมน์ Date ชัดเจน
    curve_idx = curve.copy()
    try:
        curve_idx.index = pd.to_datetime(curve_idx.index, errors="coerce")
    except Exception:
        pass
    curve_reset = curve_idx.reset_index()
    if curve_reset.columns[0] != "Date":
        curve_reset = curve_reset.rename(columns={curve_reset.columns[0]: "Date"})

    # คอลัมน์หลักที่ต้องใช้
    cols = [c for c in ["NAV", "Cash", "MV"] if c in curve_reset.columns]
    if not cols:
        st.warning("ไม่พบคอลัมน์สำหรับพล็อต (NAV/Cash/MV)")
    else:
        # ให้เป็นตัวเลข และตัด NaN ทั้งหมดออกก่อนพล็อต
        for c in cols:
            curve_reset[c] = pd.to_numeric(curve_reset[c], errors="coerce")
        plot_df = curve_reset.dropna(subset=cols, how="all").copy()

        if plot_df.empty:
            st.info("ข้อมูลเส้นว่างเปล่า (NaN ทั้งหมด) — อัปเดตราคา/ธุรกรรมให้ครบก่อน")
        else:
            # === เพิ่มคอลัมน์ Portfolio Growth (%) ===
            try:
                # เงินต้นก้อนแรก = NAV แถวแรกที่ไม่ใช่ NaN
                initial_capital = curve_idx["NAV"].dropna().iloc[0]
            except Exception:
                initial_capital = None

            if initial_capital and initial_capital != 0:
                plot_df["GrowthPct"] = (plot_df["NAV"] / float(initial_capital) - 1.0) * 100.0
            else:
                plot_df["GrowthPct"] = pd.NA  # ไม่มีเงินต้นให้คำนวณ

            # ========= พล็อตด้วยแกนคู่ =========
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # เส้นหลักซ้าย: NAV / Cash / MV (เฉพาะคอลัมน์ที่มีจริง)
            for c in cols:
                if c in plot_df.columns and plot_df[c].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df["Date"],
                            y=plot_df[c],
                            mode="lines",
                            name=c
                        ),
                        secondary_y=False
                    )

            # เส้นขวา: Portfolio Growth (%)
            if plot_df["GrowthPct"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["Date"],
                        y=plot_df["GrowthPct"],
                        mode="lines",
                        name="Growth (%)",
                        line=dict(dash="dash")
                    ),
                    secondary_y=True
                )

            # จัด labeling
            fig.update_layout(title_text="NAV / Cash / MV + Portfolio Growth (%)")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Value (฿)", secondary_y=False)
            fig.update_yaxes(title_text="Growth (%)", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("ตารางข้อมูล (ท้าย 10 แถว)"):
                cols_show = [c for c in ["NAV", "Cash", "MV", "GrowthPct"] if c in plot_df.columns]
                st.dataframe(plot_df[["Date"] + cols_show].tail(10))


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
