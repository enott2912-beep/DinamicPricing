import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app_state.db"
DEFAULT_SESSION_TTL_DAYS = 14


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db() -> None:
    with _open_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                generation_mode TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_datasets (
                session_id TEXT PRIMARY KEY,
                sales_history_csv TEXT,
                predict_csv TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_rules (
                session_id TEXT PRIMARY KEY,
                rules_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)")


def cleanup_expired_sessions() -> int:
    init_db()
    now_iso = _utc_now_iso()
    with _open_conn() as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now_iso,))
        return cursor.rowcount if cursor.rowcount is not None else 0


def _ensure_session_row(session_id: str, ttl_days: int = DEFAULT_SESSION_TTL_DAYS) -> None:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    expires_iso = (now + timedelta(days=ttl_days)).isoformat()
    with _open_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions(session_id, created_at, last_seen_at, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                expires_at=excluded.expires_at
            """,
            (session_id, now_iso, now_iso, expires_iso),
        )


def get_or_create_session_id(ttl_days: int = DEFAULT_SESSION_TTL_DAYS) -> str:
    init_db()
    if st.session_state.get("_cleanup_done") is None:
        cleanup_expired_sessions()
        st.session_state["_cleanup_done"] = True

    sid = st.session_state.get("session_id")
    if not sid:
        sid = st.query_params.get("sid")
    if not sid:
        sid = str(uuid.uuid4())

    st.session_state["session_id"] = sid
    st.query_params["sid"] = sid
    _ensure_session_row(sid, ttl_days=ttl_days)
    return sid


def get_current_session_id() -> str | None:
    return st.session_state.get("session_id")


def set_active_session_id(session_id: str, ttl_days: int = DEFAULT_SESSION_TTL_DAYS) -> str:
    init_db()
    st.session_state["session_id"] = session_id
    st.query_params["sid"] = session_id
    _ensure_session_row(session_id, ttl_days=ttl_days)
    return session_id


def start_new_session(ttl_days: int = DEFAULT_SESSION_TTL_DAYS) -> str:
    new_sid = str(uuid.uuid4())
    return set_active_session_id(new_sid, ttl_days=ttl_days)


def clear_session_data(session_id: str) -> None:
    init_db()
    _ensure_session_row(session_id)
    with _open_conn() as conn:
        conn.execute("DELETE FROM session_datasets WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_rules WHERE session_id = ?", (session_id,))
        conn.execute(
            """
            UPDATE sessions
            SET generation_mode = NULL, last_seen_at = ?, expires_at = ?
            WHERE session_id = ?
            """,
            (
                _utc_now_iso(),
                (datetime.now(timezone.utc) + timedelta(days=DEFAULT_SESSION_TTL_DAYS)).isoformat(),
                session_id,
            ),
        )


def _serialize_df(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def _deserialize_df(csv_text: str | None) -> pd.DataFrame | None:
    if not csv_text:
        return None
    from io import StringIO

    return pd.read_csv(StringIO(csv_text))


def save_sales_history_df(session_id: str, df: pd.DataFrame) -> None:
    init_db()
    _ensure_session_row(session_id)
    now_iso = _utc_now_iso()
    csv_blob = _serialize_df(df)
    with _open_conn() as conn:
        conn.execute(
            """
            INSERT INTO session_datasets(session_id, sales_history_csv, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                sales_history_csv=excluded.sales_history_csv,
                updated_at=excluded.updated_at
            """,
            (session_id, csv_blob, now_iso),
        )


def load_sales_history_df(session_id: str) -> pd.DataFrame | None:
    init_db()
    _ensure_session_row(session_id)
    with _open_conn() as conn:
        row = conn.execute(
            "SELECT sales_history_csv FROM session_datasets WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return _deserialize_df(row[0]) if row else None


def clear_predict_df(session_id: str, columns: list[str]) -> None:
    init_db()
    _ensure_session_row(session_id)
    now_iso = _utc_now_iso()
    empty_csv = pd.DataFrame(columns=columns).to_csv(index=False)
    with _open_conn() as conn:
        conn.execute(
            """
            INSERT INTO session_datasets(session_id, predict_csv, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                predict_csv=excluded.predict_csv,
                updated_at=excluded.updated_at
            """,
            (session_id, empty_csv, now_iso),
        )


def save_predict_df(session_id: str, df: pd.DataFrame) -> None:
    init_db()
    _ensure_session_row(session_id)
    now_iso = _utc_now_iso()
    csv_blob = _serialize_df(df)
    with _open_conn() as conn:
        conn.execute(
            """
            INSERT INTO session_datasets(session_id, predict_csv, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                predict_csv=excluded.predict_csv,
                updated_at=excluded.updated_at
            """,
            (session_id, csv_blob, now_iso),
        )


def load_predict_df(session_id: str) -> pd.DataFrame | None:
    init_db()
    _ensure_session_row(session_id)
    with _open_conn() as conn:
        row = conn.execute(
            "SELECT predict_csv FROM session_datasets WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return _deserialize_df(row[0]) if row else None


def set_generation_mode(session_id: str, mode: str) -> None:
    init_db()
    _ensure_session_row(session_id)
    now_iso = _utc_now_iso()
    expires_iso = (datetime.now(timezone.utc) + timedelta(days=DEFAULT_SESSION_TTL_DAYS)).isoformat()
    with _open_conn() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET generation_mode = ?, last_seen_at = ?, expires_at = ?
            WHERE session_id = ?
            """,
            (mode, now_iso, expires_iso, session_id),
        )


def get_generation_mode(session_id: str) -> str | None:
    init_db()
    _ensure_session_row(session_id)
    with _open_conn() as conn:
        row = conn.execute(
            "SELECT generation_mode FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return row[0] if row[0] else None


def save_rules(session_id: str, rules: list[dict]) -> None:
    init_db()
    _ensure_session_row(session_id)
    now_iso = _utc_now_iso()
    payload = json.dumps(rules, ensure_ascii=False, indent=2)
    with _open_conn() as conn:
        conn.execute(
            """
            INSERT INTO session_rules(session_id, rules_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                rules_json=excluded.rules_json,
                updated_at=excluded.updated_at
            """,
            (session_id, payload, now_iso),
        )


def load_rules(session_id: str) -> list[dict] | None:
    init_db()
    _ensure_session_row(session_id)
    with _open_conn() as conn:
        row = conn.execute(
            "SELECT rules_json FROM session_rules WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None
