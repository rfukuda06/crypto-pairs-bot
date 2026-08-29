# src/pairsbot/storage.py
from __future__ import annotations

import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    pair TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    run_id INTEGER, ts TEXT, equity REAL,
    UNIQUE(run_id, ts)
);
CREATE TABLE IF NOT EXISTS trades (
    run_id INTEGER, ts TEXT, symbol TEXT, notional REAL,
    price REAL, qty REAL, fee REAL, reason TEXT,
    UNIQUE(run_id, ts, symbol)
);
CREATE TABLE IF NOT EXISTS positions (
    run_id INTEGER, symbol TEXT, qty REAL, avg_price REAL,
    PRIMARY KEY (run_id, symbol)
);
"""


class Store:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_run(self, mode: str, pair: str | None) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (mode, pair) VALUES (?, ?)", (mode, pair))
        self._conn.commit()
        return int(cur.lastrowid)

    def record_equity(self, run_id: int, ts: str, equity: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO equity (run_id, ts, equity) VALUES (?, ?, ?)",
            (run_id, ts, equity))
        self._conn.commit()

    def record_trade(self, run_id: int, ts: str, symbol: str, notional: float,
                     price: float, qty: float, fee: float, reason: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trades "
            "(run_id, ts, symbol, notional, price, qty, fee, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, ts, symbol, notional, price, qty, fee, reason))
        self._conn.commit()

    def upsert_position(self, run_id: int, symbol: str, qty: float,
                        avg_price: float) -> None:
        self._conn.execute(
            "INSERT INTO positions (run_id, symbol, qty, avg_price) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(run_id, symbol) DO UPDATE SET qty=excluded.qty, "
            "avg_price=excluded.avg_price",
            (run_id, symbol, qty, avg_price))
        self._conn.commit()

    def load_equity(self, run_id: int) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT ts, equity FROM equity WHERE run_id=? ORDER BY ts", (run_id,))
        return [(ts, eq) for ts, eq in rows]

    def load_trades(self, run_id: int) -> list[dict]:
        cur = self._conn.execute(
            "SELECT ts, symbol, notional, price, qty, fee, reason "
            "FROM trades WHERE run_id=? ORDER BY ts", (run_id,))
        cols = ["ts", "symbol", "notional", "price", "qty", "fee", "reason"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_run(self, run_id: int | None = None) -> tuple[int, str, str | None] | None:
        if run_id is None:
            # id is AUTOINCREMENT, so the highest id is the most recent run.
            row = self._conn.execute(
                "SELECT id, mode, pair FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        else:
            row = self._conn.execute(
                "SELECT id, mode, pair FROM runs WHERE id=?", (run_id,)).fetchone()
        return (int(row[0]), row[1], row[2]) if row else None

    def latest_live_run(self, pair: str) -> int | None:
        """Most recent live run for a pair, or None. Used to resume on restart."""
        row = self._conn.execute(
            "SELECT id FROM runs WHERE mode='live' AND pair=? ORDER BY id DESC LIMIT 1",
            (pair,)).fetchone()
        return int(row[0]) if row else None

    def load_positions(self, run_id: int) -> dict[str, tuple[float, float]]:
        rows = self._conn.execute(
            "SELECT symbol, qty, avg_price FROM positions WHERE run_id=? AND qty != 0",
            (run_id,))
        return {sym: (qty, avg) for sym, qty, avg in rows}
