import sqlite3
import contextlib
import os
from pathlib import Path

# Fix 4: configurable path for seen-files registry
_DB_PATH: str = os.environ.get(
    "SEEN_FILES_DB",
    "data/processed/seen_files.db")

def _conn():
    db = Path(_DB_PATH)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS seen_files (
            fits_path       TEXT PRIMARY KEY,
            instrument      TEXT NOT NULL,
            processed_at    TEXT NOT NULL DEFAULT (datetime('now')),
            pipeline_run_id TEXT NOT NULL
        )""")
    con.commit()
    return con

def is_seen(fits_path: str) -> bool:
    '''Return True if this FITS file has already been ingested.'''
    with contextlib.closing(_conn()) as con:
        row = con.execute(
            "SELECT 1 FROM seen_files WHERE fits_path=?",
            (fits_path,)).fetchone()
    return row is not None

def mark_seen(fits_path: str, instrument: str,
              pipeline_run_id: str) -> None:
    '''Register a FITS file as processed. Idempotent (IGNORE on dup).'''
    with contextlib.closing(_conn()) as con:
        con.execute(
            "INSERT OR IGNORE INTO seen_files "
            "(fits_path, instrument, pipeline_run_id) VALUES (?,?,?)",
            (fits_path, instrument, pipeline_run_id))
        con.commit()

def list_seen(instrument: str | None = None) -> list[dict]:
    '''Audit: return all seen files, optionally filtered by instrument.'''
    with contextlib.closing(_conn()) as con:
        q = "SELECT * FROM seen_files"
        if instrument:
            q += " WHERE instrument=?"
            rows = con.execute(q, (instrument,)).fetchall()
        else:
            rows = con.execute(q).fetchall()
    keys = ["fits_path","instrument","processed_at","pipeline_run_id"]
    return [dict(zip(keys, r)) for r in rows]
