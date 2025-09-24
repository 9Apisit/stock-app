# db_utils.py
# ================== ยูทิลฐานข้อมูล SQLite แบบรวมศูนย์ ======================
# ส่วนที่เพิ่มในสเต็ปนี้:
#   - ตาราง cash_ledger สำหรับฝาก/ถอน/ปรับยอด/ค่าธรรมเนียมเงินสด
#   - ฟังก์ชัน add_cash, read_cash_ledger, compute_cash_balance
# หมายเหตุ: ยังไม่แตะหน้า UI; เอาไว้สเต็ปถัดไป

import sqlite3
from typing import Optional, Tuple, List
import pandas as pd

DB_PATH = "portfolio.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS portfolios (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS symbols (
  id     INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL UNIQUE,
  note   TEXT
);

CREATE TABLE IF NOT EXISTS trades (
  id           INTEGER PRIMARY KEY,
  portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  symbol_id    INTEGER NOT NULL REFERENCES symbols(id),
  ts           TEXT NOT NULL,  -- YYYY-MM-DD
  side         TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  qty          INTEGER NOT NULL CHECK (qty > 0),
  price        REAL NOT NULL CHECK (price >= 0),
  fee          REAL NOT NULL DEFAULT 0.0,
  UNIQUE(portfolio_id, symbol_id, ts, side, qty, price, fee)
);

CREATE TABLE IF NOT EXISTS prices (
  symbol_id  INTEGER NOT NULL REFERENCES symbols(id),
  d          TEXT NOT NULL,  -- YYYY-MM-DD
  close      REAL NOT NULL,
  source     TEXT DEFAULT 'yfinance',
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol_id, d)
);

-- ================== ตารางใหม่: cash_ledger ==================
-- เก็บธุรกรรมเงินสดของแต่ละพอร์ต (ฝาก/ถอน/ปรับยอด/ค่าธรรมเนียมเงินสด)
-- สัญญะ: DEPOSIT/ADJUST = จำนวนบวก, WITHDRAW/FEE = จำนวนลบ
CREATE TABLE IF NOT EXISTS cash_ledger (
  id           INTEGER PRIMARY KEY,
  portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  ts           TEXT NOT NULL,   -- YYYY-MM-DD
  type         TEXT NOT NULL CHECK (type IN ('DEPOSIT','WITHDRAW','ADJUST','FEE')),
  amount       REAL NOT NULL,   -- บวก/ลบตาม type
  note         TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_by_symbol_date ON trades(symbol_id, ts);
CREATE INDEX IF NOT EXISTS idx_prices_by_symbol_date ON prices(symbol_id, d);
CREATE INDEX IF NOT EXISTS idx_cash_by_pf_ts        ON cash_ledger(portfolio_id, ts);
"""

# ------------------------- Core connection -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

# ------------------------- Portfolio / Symbol ----------------------
def ensure_portfolio(name: str) -> int:
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO portfolios(name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM portfolios WHERE name=?", (name,)).fetchone()
    return int(row[0])

def upsert_symbol(symbol: str) -> int:
    sym = symbol.strip().upper()
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO symbols(symbol) VALUES (?)", (sym,))
        row = conn.execute("SELECT id FROM symbols WHERE symbol=?", (sym,)).fetchone()
    return int(row[0])

# ------------------------- Trades (หุ้น) ---------------------------
def add_trade(pf_id: int, symbol: str, ts: str, side: str, qty: int, price: float, fee: float = 0.0):
    sym_id = upsert_symbol(symbol)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO trades(portfolio_id, symbol_id, ts, side, qty, price, fee)
            VALUES (?,?,?,?,?,?,?)
            """,
            (pf_id, sym_id, ts, side, qty, price, fee),
        )

def read_trades_df(pf_id: int, symbol: Optional[str] = None) -> pd.DataFrame:
    q = """
        SELECT t.id, p.name AS portfolio, s.symbol, t.ts, t.side, t.qty, t.price, t.fee
        FROM trades t
        JOIN portfolios p ON p.id = t.portfolio_id
        JOIN symbols    s ON s.id = t.symbol_id
        WHERE t.portfolio_id = ?
    """
    params = [pf_id]
    if symbol:
        q += " AND s.symbol = ?"
        params.append(symbol.upper().strip())
    q += " ORDER BY t.ts, t.id"
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=params)
    return df

def read_all_portfolios() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("SELECT id, name FROM portfolios ORDER BY name", conn)
    return df

def read_symbols_for_portfolio(pf_id: int) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT s.symbol
            FROM trades t JOIN symbols s ON s.id=t.symbol_id
            WHERE t.portfolio_id=?
            ORDER BY s.symbol
        """, (pf_id,)).fetchall()
    return [r[0] for r in rows]

def get_current_qty(pf_id: int, symbol: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(
                SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END), 0
            )
            FROM trades t
            JOIN symbols s ON s.id = t.symbol_id
            WHERE t.portfolio_id=? AND s.symbol=?;
            """,
            (pf_id, symbol.upper().strip()),
        ).fetchone()
    return int(row[0] if row and row[0] is not None else 0)

def safe_add_trade(pf_id: int, symbol: str, ts: str, side: str, qty: int, price: float, fee: float = 0.0) -> str:
    """
    ตรวจสอบก่อนบันทึกธุรกรรม:
      - SELL: ต้องมีของพอขาย
      - BUY : ต้องมีเงินสด (Cash Balance) พอจ่าย = qty*price + fee
    ผ่านแล้วจึง add_trade(...)
    """
    symbol = symbol.strip().upper()
    qty = int(qty)
    price = float(price)
    fee = float(fee)

    if side == "SELL":
        on_hand = get_current_qty(pf_id, symbol)
        if qty > on_hand:
            return f"❌ ขายเกินจำนวนที่ถืออยู่ (ถืออยู่ {on_hand}, จะขาย {qty})"

    elif side == "BUY":
        # ดึงเงินสดปัจจุบัน (รวมผลจาก cash ledger และเทรดที่มีอยู่แล้ว)
        try:
            cash = compute_cash_balance(pf_id)
        except NameError:
            # เผื่อโปรเจกต์คุณยังไม่มี compute_cash_balance ให้ดึงจาก cash_series แทน
            cash_series = compute_cash_series(pf_id)
            cash = float(cash_series.iloc[-1]) if cash_series is not None and not cash_series.empty else 0.0

        need = qty * price + fee
        # เผื่อ floating error เล็กน้อย
        if need > cash + 1e-6:
            return f"❌ เงินสดไม่พอ ซื้อ {symbol} ต้องใช้ {need:,.2f} แต่มี {cash:,.2f} (ฝากเงินเพิ่ม หรือขายสินทรัพย์ก่อน)"

    # ผ่านเงื่อนไข → บันทึกธุรกรรม
    add_trade(pf_id, symbol, ts, side, qty, price, fee)
    return "✅ บันทึกธุรกรรมแล้ว"

# ------------------------- Prices (แคชราคา) -----------------------
def upsert_prices(symbol: str, prices_df: pd.DataFrame):
    if prices_df is None or prices_df.empty:
        return
    sym_id = upsert_symbol(symbol)
    rows = []
    for d, c in prices_df[["Date", "Close"]].itertuples(index=False):
        if pd.isna(c):
            continue
        rows.append((sym_id, pd.to_datetime(d).strftime("%Y-%m-%d"), float(c)))
    if not rows:
        return
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO prices(symbol_id, d, close)
            VALUES (?,?,?)
            ON CONFLICT(symbol_id, d) DO UPDATE SET
                close = excluded.close,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )

def load_prices_df(symbol: str) -> pd.DataFrame:
    sym_id = upsert_symbol(symbol)
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT d AS Date, close AS Close FROM prices WHERE symbol_id=? ORDER BY d",
            conn, params=(sym_id,), parse_dates=["Date"],
        )
    return df

# ------------------------- P&L (avg cost) -------------------------
def average_cost_position(trades_df: pd.DataFrame) -> Tuple[int, float, float]:
    """คำนวณคงเหลือ/ต้นทุนเฉลี่ย/กำไรรับรู้สะสม แบบ Average Cost"""
    if trades_df is None or trades_df.empty:
        return 0, 0.0, 0.0
    qty, avg_cost, realized = 0, 0.0, 0.0
    for _, r in trades_df.sort_values("ts").iterrows():
        side, q, px, fee = r["side"], int(r["qty"]), float(r["price"]), float(r["fee"])
        if side == "BUY":
            new_qty = qty + q
            avg_cost = (qty * avg_cost + q * (px + fee / max(1, q))) / max(1, new_qty)
            qty = new_qty
        else:
            realized += (px - avg_cost) * q - fee
            qty -= q
            if qty == 0:
                avg_cost = 0.0
    return qty, avg_cost, realized

# ========================= NEW: Cash Ledger =========================
def add_cash(pf_id: int, ts: str, type_: str, amount: float, note: Optional[str] = None):
    """
    บันทึกธุรกรรมเงินสดของพอร์ต
    - type_: 'DEPOSIT','WITHDRAW','ADJUST','FEE'
    - amount: แนะนำใส่เครื่องหมายตาม type (DEPOSIT/ADJUST บวก, WITHDRAW/FEE ลบ)
      ถ้าใส่ค่าบวกเสมอ โค้ดนี้จะจัดสัญญะให้เอง
    """
    type_ = type_.upper().strip()
    if type_ not in ("DEPOSIT", "WITHDRAW", "ADJUST", "FEE"):
        raise ValueError("type ต้องเป็น DEPOSIT/WITHDRAW/ADJUST/FEE")

    # ทำให้สัญญะ amount สอดคล้องกับ type
    amt = float(amount)
    if type_ in ("WITHDRAW", "FEE") and amt > 0:
        amt = -amt
    if type_ in ("DEPOSIT", "ADJUST") and amt < 0:
        amt = -amt

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO cash_ledger(portfolio_id, ts, type, amount, note)
            VALUES (?,?,?,?,?)
            """,
            (pf_id, ts, type_, amt, note),
        )

def read_cash_ledger(pf_id: int,
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None) -> pd.DataFrame:
    """
    ดึงรายการเงินสดตามพอร์ต (ช่วงวันเลือกได้, ปล่อยว่างคือทั้งหมด)
    """
    q = """
        SELECT id, portfolio_id, ts, type, amount, note
        FROM cash_ledger
        WHERE portfolio_id = ?
    """
    params: List = [pf_id]
    if date_from:
        q += " AND ts >= ?"
        params.append(date_from)
    if date_to:
        q += " AND ts <= ?"
        params.append(date_to)
    q += " ORDER BY ts, id"
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=params)
    return df

def compute_cash_balance(pf_id: int) -> float:
    """
    คำนวณเงินสดคงเหลือปัจจุบันจาก:
      1) ผลรวม cash_ledger.amount
      2) กระแสเงินจาก trades:
         - BUY: ออก = (qty*price + fee)
         - SELL: เข้า = (qty*price - fee)
    สูตร:
      Cash = Σ(ledger) - Σ(BUY_value+fee) + Σ(SELL_value-fee)
    """
    with get_conn() as conn:
        # 1) ledger
        row1 = conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) FROM cash_ledger WHERE portfolio_id=?",
            (pf_id,)
        ).fetchone()
        ledger_sum = float(row1[0] if row1 and row1[0] is not None else 0.0)

        # 2) trades outflow (BUY) และ inflow (SELL)
        row2 = conn.execute("""
            SELECT
              COALESCE(SUM(CASE WHEN side='BUY'  THEN (qty*price + fee) END), 0.0) AS buy_out,
              COALESCE(SUM(CASE WHEN side='SELL' THEN (qty*price - fee) END), 0.0) AS sell_in
            FROM trades
            WHERE portfolio_id=?
        """, (pf_id,)).fetchone()
        buy_out  = float(row2[0] if row2 and row2[0] is not None else 0.0)
        sell_in  = float(row2[1] if row2 and row2[1] is not None else 0.0)

    cash = ledger_sum - buy_out + sell_in
    return float(cash)

# -------- ฟังก์ชันช่วยในสเต็ปถัดไป (เตรียมไว้ล่วงหน้า) --------
def get_initial_deposit(pf_id: int) -> Optional[Tuple[str, float]]:
    """
    คืน (วันที่, จำนวน) ของ DEPOSIT แรกของพอร์ต ถ้าไม่มีคืน None
    ใช้เป็นฐานคำนวณ % เติบโต (โจทย์หน้า Dashboard)
    """
    with get_conn() as conn:
        row = conn.execute("""
            SELECT ts, amount
            FROM cash_ledger
            WHERE portfolio_id=? AND type='DEPOSIT'
            ORDER BY ts, id
            LIMIT 1
        """, (pf_id,)).fetchone()
    if not row:
        return None
    return str(row[0]), float(row[1])

# ========================= STEP 2: NAV & EQUITY CURVE =========================
# แนวคิด:
# - เงินสดรายวัน: สร้าง time series จาก cash_ledger และกระแสเงินซื้อ/ขายหุ้น แล้วทำ cumulative sum
# - ปริมาณหุ้นรายวัน: สร้าง time series สะสมจาก trades ต่อสัญลักษณ์
# - ราคาหุ้นรายวัน: ดึงจากตาราง prices แล้วปรับเป็นวันทำการ (B) + ffill
# - NAV(t) = Cash(t) + Σ [Qty_i(t) × Price_i(t)]
# - เริ่มช่วงเวลาอย่างน้อยตั้งแต่ initial deposit ตัวแรกของพอร์ต

import pandas as _pd
from datetime import datetime as _dt, date as _date

def _to_date_str(d) -> str:
    """รับ str/datetime/date แล้วคืนเป็น 'YYYY-MM-DD'"""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, _pd.Timestamp):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, _dt):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, _date):
        return d.strftime("%Y-%m-%d")
    raise ValueError("unsupported date type")

def _biz_range(start: str, end: str) -> _pd.DatetimeIndex:
    s = _pd.to_datetime(start)
    e = _pd.to_datetime(end)
    # อย่างน้อยต้องมี 1 วัน
    if e < s:
        e = s
    return _pd.bdate_range(s, e, freq="B")

def _get_analysis_window(pf_id: int, start: str | None, end: str | None):
    """กำหนดช่วงวันที่สำหรับคำนวณ: จาก initial deposit ตัวแรก → วันนี้ (หรือ end ที่ส่งมา)"""
    today = _pd.Timestamp.today().normalize()
    init = get_initial_deposit(pf_id)
    if init:
        start0 = _pd.to_datetime(init[0])
    else:
        # ถ้าไม่มี DEPOSIT ให้เริ่มจากวันแรกที่มี trade หรือ cash ใด ๆ
        with get_conn() as conn:
            row = conn.execute("""
                SELECT MIN(dt) FROM (
                  SELECT MIN(ts) AS dt FROM trades WHERE portfolio_id=?
                  UNION ALL
                  SELECT MIN(ts) AS dt FROM cash_ledger WHERE portfolio_id=?
                )
            """, (pf_id, pf_id)).fetchone()
        start0 = _pd.to_datetime(row[0]) if row and row[0] else today

    start_dt = _pd.to_datetime(start) if start else start0
    end_dt   = _pd.to_datetime(end)   if end   else today
    return _to_date_str(start_dt), _to_date_str(end_dt)

def compute_cash_series(pf_id: int, start: str | None = None, end: str | None = None) -> _pd.Series:
    """
    คืนซีรีส์เงินสดสะสมรายวัน (index=วันทำการ, ชื่อคอลัมน์='Cash')
    สูตรเงินสด:
      Cash = Σ(ledger.amount) - Σ(BUY_value+fee) + Σ(SELL_value-fee)  [สะสมตามวัน]
    """
    start_str, end_str = _get_analysis_window(pf_id, start, end)
    idx = _biz_range(start_str, end_str)
    if len(idx) == 0:
        return _pd.Series([], name="Cash", dtype=float)

    # 1) เงินสดจาก cash_ledger
    ledger = read_cash_ledger(pf_id)
    if ledger is None or ledger.empty:
        ledger_daily = _pd.Series(0.0, index=idx)
    else:
        ledger = ledger.copy()
        ledger["ts"] = _pd.to_datetime(ledger["ts"])
        ledger = ledger[(ledger["ts"] >= idx[0]) & (ledger["ts"] <= idx[-1])]
        ledger_daily = ledger.groupby(ledger["ts"].dt.normalize())["amount"].sum()
        ledger_daily = ledger_daily.reindex(idx, fill_value=0.0)

    # 2) กระแสเงินจาก trades (BUY ออกเงิน, SELL เข้าเงิน)
    trades = read_trades_df(pf_id)
    if trades is None or trades.empty:
        trade_flow = _pd.Series(0.0, index=idx)
    else:
        t = trades.copy()
        t["ts"] = _pd.to_datetime(t["ts"])
        t = t[(t["ts"] >= idx[0]) & (t["ts"] <= idx[-1])]

        # ✅ เริ่มจาก 0.0 แทนการใช้ pd.NA (หลบ TypeError บน pandas 2.x / Python 3.13)
        flow = _pd.Series(0.0, index=t.index, dtype="float64")

        buy_mask = t["side"].eq("BUY")
        sell_mask = t["side"].eq("SELL")

        if buy_mask.any():
            flow.loc[buy_mask] = -(
                t.loc[buy_mask, "qty"] * t.loc[buy_mask, "price"] + t.loc[buy_mask, "fee"]
            ).astype("float64")

        if sell_mask.any():
            flow.loc[sell_mask] = +(
                t.loc[sell_mask, "qty"] * t.loc[sell_mask, "price"] - t.loc[sell_mask, "fee"]
            ).astype("float64")

        flow_by_day = flow.groupby(t["ts"].dt.normalize()).sum()
        trade_flow = flow_by_day.reindex(idx, fill_value=0.0)

    daily_net = ledger_daily.add(trade_flow, fill_value=0.0)
    cash_series = daily_net.cumsum()
    cash_series.name = "Cash"
    return cash_series

def _position_qty_series_for_symbol(pf_id: int, symbol: str, start: str, end: str) -> _pd.Series:
    """คืนซีรีส์จำนวนคงเหลือรายวันของ symbol นั้น ๆ (สะสมจาก BUY/SELL)"""
    idx = _biz_range(start, end)
    trades = read_trades_df(pf_id, symbol=symbol)
    if trades is None or trades.empty:
        return _pd.Series(0, index=idx, name=symbol, dtype="int64")
    t = trades.copy()
    t["ts"] = _pd.to_datetime(t["ts"])
    t = t[(t["ts"] >= idx[0]) & (t["ts"] <= idx[-1])]
    if t.empty:
        return _pd.Series(0, index=idx, name=symbol, dtype="int64")
    delta = _pd.Series(0, index=idx, dtype="int64")
    buy_mask = t["side"] == "BUY"
    sell_mask = ~buy_mask
    # เปลี่ยนเป็น +qty / -qty
    delta_buy  = t.loc[buy_mask].groupby(t.loc[buy_mask, "ts"].dt.normalize())["qty"].sum()
    delta_sell = t.loc[sell_mask].groupby(t.loc[sell_mask, "ts"].dt.normalize())["qty"].sum() * -1
    if not delta_buy.empty:
        delta = delta.add(delta_buy.reindex(idx, fill_value=0), fill_value=0)
    if not delta_sell.empty:
        delta = delta.add(delta_sell.reindex(idx, fill_value=0), fill_value=0)
    qty_series = delta.cumsum()
    qty_series.name = symbol
    return qty_series

def _price_series_for_symbol(symbol: str, start: str, end: str) -> _pd.Series:
    """คืนซีรีส์ราคาปิดรายวัน (Close) ของ symbol ในช่วงวันทำการ; ถ้าไม่มีให้คืน series ว่าง"""
    prices = load_prices_df(symbol)
    if prices is None or prices.empty:
        return _pd.Series(dtype="float64", name=symbol)
    s = prices.copy()
    s["Date"] = _pd.to_datetime(s["Date"])
    s = s[(s["Date"] >= start) & (s["Date"] <= end)]
    if s.empty:
        # อนุโลม: ลองใช้ข้อมูลทั้งหมดแล้วค่อยรีอินเด็กซ์ช่วง
        s = prices.copy()
        s["Date"] = _pd.to_datetime(s["Date"])
    s = s.sort_values("Date").set_index("Date")["Close"].astype(float)
    idx = _biz_range(start, end)
    s = s.reindex(idx, method="ffill")  # เติมวันหยุด/ขาดข้อมูล
    s.name = symbol
    return s

def compute_equity_curve(pf_id: int, symbols: list[str] | None = None,
                         start: str | None = None, end: str | None = None) -> _pd.DataFrame:
    """
    คืน DataFrame: index=วันทำการ, columns=['NAV','Cash','MV'] และ (ถ้าต้องการ) MV ราย symbol (เช่น 'MV:AAPL')
    ข้อกำหนด:
      - ต้องมีราคาในตาราง prices สำหรับหุ้นที่ถืออยู่ (อย่างน้อยบางวัน) โค้ดจะ ffill ให้
    """
    start_str, end_str = _get_analysis_window(pf_id, start, end)
    idx = _biz_range(start_str, end_str)
    if len(idx) == 0:
        return _pd.DataFrame()

    # 1) เงินสดสะสมรายวัน
    cash = compute_cash_series(pf_id, start_str, end_str)

    # 2) สัญลักษณ์ที่ต้องคำนวณ
    if symbols is None:
        symbols = read_symbols_for_portfolio(pf_id)
    symbols = list(sorted(set(symbols)))
    # 3) Qty series + Price series + MV
    mv_total = _pd.Series(0.0, index=idx, name="MV")
    mv_by_symbol = {}
    for sym in symbols:
        qty = _position_qty_series_for_symbol(pf_id, sym, start_str, end_str)
        if qty.abs().sum() == 0:
            continue  # ไม่ถือเลยทั้งช่วง
        px = _price_series_for_symbol(sym, start_str, end_str)
        if px.empty:
            # ถ้าไม่มีราคาเลย ให้ข้ามสัญลักษณ์นี้ (หรือจะถือว่าราคาเท่า avg_cost ก็ได้ แต่เวอร์ชันแรกขอข้าม)
            continue
        mv_sym = (qty.astype(float) * px.astype(float))
        mv_total = mv_total.add(mv_sym, fill_value=0.0)
        mv_by_symbol[sym] = mv_sym

    nav = cash.add(mv_total, fill_value=0.0)
    out = _pd.DataFrame({"Cash": cash, "MV": mv_total, "NAV": nav})

    # แนบ MV ต่อ symbol ไว้ดูประกอบ (คอลัมน์ชื่อ 'MV:<SYM>')
    for sym, s in mv_by_symbol.items():
        out[f"MV:{sym}"] = s

    return out

def compute_nav_now(pf_id: int) -> dict:
    """
    สรุปสถานะปัจจุบัน:
      - cash_now
      - mv_now (รวมมูลค่าตลาดของสถานะที่ถือ)
      - nav_now
      - realized_pnl, unrealized_pnl (รวมทุกสัญลักษณ์)
    หมายเหตุ: ใช้ 'ราคาวันล่าสุดที่มีในตาราง prices' (ผู้ใช้ควรอัปเดตราคาให้ทันสมัยในแอพหน้า 1)
    """
    cash_now = compute_cash_balance(pf_id)

    symbols = read_symbols_for_portfolio(pf_id)
    realized_total = 0.0
    unreal_total = 0.0
    mv = 0.0

    for sym in symbols:
        tdf = read_trades_df(pf_id, sym)
        qty, avg_cost, realized = average_cost_position(tdf)
        realized_total += realized
        # ราคาล่าสุดจากตาราง prices
        px_df = load_prices_df(sym)
        last = float(px_df["Close"].iloc[-1]) if not px_df.empty else avg_cost
        mv += qty * last
        unreal_total += (last - avg_cost) * qty

    nav_now = cash_now + mv
    return {
        "cash_now": float(cash_now),
        "mv_now": float(mv),
        "nav_now": float(nav_now),
        "realized_pnl": float(realized_total),
        "unrealized_pnl": float(unreal_total),
    }

def compute_growth_vs_initial(pf_id: int) -> dict:
    """
    คืน % เติบโตเมื่อเทียบกับเงินก้อนแรก (Initial_Deposit)
    - ถ้าไม่มี DEPOSIT เลย จะคืน growth_pct=None
    """
    snap = compute_nav_now(pf_id)
    init = get_initial_deposit(pf_id)
    if not init:
        return {"initial_deposit": None, "growth_pct": None, "nav_now": snap["nav_now"]}
    initial_amt = float(init[1])
    if initial_amt <= 0:
        return {"initial_deposit": initial_amt, "growth_pct": None, "nav_now": snap["nav_now"]}
    growth_pct = (snap["nav_now"] - initial_amt) / initial_amt * 100.0
    return {"initial_deposit": initial_amt, "growth_pct": float(growth_pct), "nav_now": snap["nav_now"]}
