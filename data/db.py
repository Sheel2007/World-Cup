"""SQLite schema + connection helpers."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    tournament TEXT,
    city TEXT,
    country TEXT,
    neutral INTEGER NOT NULL DEFAULT 0,
    UNIQUE(date, home_team, away_team)
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team);
CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team);

CREATE TABLE IF NOT EXISTS elo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    date TEXT NOT NULL,
    rating REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_elo_team ON elo(team);
CREATE INDEX IF NOT EXISTS idx_elo_date ON elo(date);

CREATE TABLE IF NOT EXISTS features (
    match_id INTEGER PRIMARY KEY,
    feature_json TEXT NOT NULL,
    label INTEGER NOT NULL,
    FOREIGN KEY(match_id) REFERENCES matches(id)
);
"""


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
