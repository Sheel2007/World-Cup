"""Fetch international results CSV and load into SQLite."""
import io
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_CSV_URL, DATA_DIR, MIN_TRAIN_YEAR
from data.db import init_db, get_conn


def fetch_results_csv(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "results.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    print(f"Fetching {RESULTS_CSV_URL}")
    r = requests.get(RESULTS_CSV_URL, timeout=60)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content))


def load_matches(df: pd.DataFrame) -> int:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["neutral"] = df["neutral"].astype(int)
    df = df[df["date"] >= f"{MIN_TRAIN_YEAR}-01-01"]
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    rows = df[
        ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament", "city", "country", "neutral"]
    ].itertuples(index=False, name=None)

    init_db()
    inserted = 0
    with get_conn() as conn:
        cur = conn.cursor()
        for row in rows:
            try:
                cur.execute(
                    """INSERT OR IGNORE INTO matches
                       (date, home_team, away_team, home_score, away_score,
                        tournament, city, country, neutral)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
                inserted += cur.rowcount
            except sqlite3.IntegrityError:
                pass
        conn.commit()
    return inserted


def main(force: bool = False):
    df = fetch_results_csv(force=force)
    n = load_matches(df)
    print(f"Inserted {n} matches from {MIN_TRAIN_YEAR} onwards.")
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        print(f"Total matches in DB: {total}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
