# db_utils.py
import sqlite3
from typing import Optional, Tuple, List
import pandas as pd

DB_PATH = "portfolio.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS portfolios (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS symbols (id INTEGER PRIMARY KEY, symbol TEXT NOT NULL UNIQUE, note TEXT);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY,
  portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  symbol_id INTEGER NOT NULL REFERENCES symbols(id),
  ts TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  qty INTEGER NOT NULL CHECK (qty > 0),
  price REAL NOT NULL CHECK (price >= 0),
  fee REAL NOT NULL DEFAULT 0.0,
  UNIQUE(portfolio_id, symbol_id, ts, side, qty, price, fee)
);
CREATE TABLE IF NOT EXISTS prices (
  symbol_id INTEGER NOT NULL REFERENCES symbols(id),
  d TEXT NOT NULL,
  close REAL NOT NULL,
  source TEXT DEFAULT 'yfinance',
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol_id, d)
);
CREATE INDEX IF NOT EXISTS idx_trades_by_symbol_date ON trades(symbol_id, ts);
CREATE INDEX IF NOT EXISTS idx_prices_by_symbol_date ON prices(symbol_id, d);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

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

def add_trade(pf_id: int, symbol: str, ts: str, side: str, qty: int, price: float, fee: float = 0.0):
    sym_id = upsert_symbol(symbol)
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO trades(portfolio_id, symbol_id, ts, side, qty, price, fee) VALUES (?,?,?,?,?,?,?)",
                     (pf_id, sym_id, ts, side, qty, price, fee))

def read_trades_df(pf_id: int, symbol: Optional[str] = None) -> pd.DataFrame:
    q = "SELECT t.id, p.name AS portfolio, s.symbol, t.ts, t.side, t.qty, t.price, t.fee FROM trades t JOIN portfolios p ON p.id = t.portfolio_id JOIN symbols s ON s.id = t.symbol_id WHERE t.portfolio_id = ?"
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
        rows = conn.execute("SELECT DISTINCT s.symbol FROM trades t JOIN symbols s ON s.id=t.symbol_id WHERE t.portfolio_id=? ORDER BY s.symbol", (pf_id,)).fetchall()
    return [r[0] for r in rows]

def get_current_qty(pf_id: int, symbol: str) -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END), 0) FROM trades t JOIN symbols s ON s.id = t.symbol_id WHERE t.portfolio_id=? AND s.symbol=?;", (pf_id, symbol.upper().strip())).fetchone()
    return int(row[0] if row and row[0] is not None else 0)

def safe_add_trade(pf_id: int, symbol: str, ts: str, side: str, qty: int, price: float, fee: float = 0.0) -> str:
    if side == "SELL":
        on_hand = get_current_qty(pf_id, symbol)
        if qty > on_hand:
            return f"❌ ขายเกินจำนวนที่ถืออยู่ (ถืออยู่ {on_hand}, จะขาย {qty})"
    add_trade(pf_id, symbol, ts, side, qty, price, fee)
    return "✅ บันทึกธุรกรรมแล้ว"

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
        conn.executemany("INSERT INTO prices(symbol_id, d, close) VALUES (?,?,?) ON CONFLICT(symbol_id, d) DO UPDATE SET close = excluded.close, updated_at = CURRENT_TIMESTAMP", rows)

def load_prices_df(symbol: str) -> pd.DataFrame:
    sym_id = upsert_symbol(symbol)
    with get_conn() as conn:
        df = pd.read_sql("SELECT d AS Date, close AS Close FROM prices WHERE symbol_id=? ORDER BY d", conn, params=(sym_id,), parse_dates=["Date"])
    return df

def average_cost_position(trades_df: pd.DataFrame) -> Tuple[int, float, float]:
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
