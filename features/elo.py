"""Compute time-series ELO ratings from match history."""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ELO_K, ELO_INITIAL, ELO_HOME_ADV
from data.db import get_conn, init_db


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def actual_score(home_goals: int, away_goals: int, perspective: str = "home") -> float:
    if home_goals > away_goals:
        return 1.0 if perspective == "home" else 0.0
    if home_goals < away_goals:
        return 0.0 if perspective == "home" else 1.0
    return 0.5


def tournament_multiplier(tournament: str) -> float:
    t = (tournament or "").lower()
    if "world cup" in t and "qualif" not in t:
        return 1.6
    if "uefa euro" in t or "copa am" in t or "africa cup" in t or "asian cup" in t:
        return 1.4
    if "qualif" in t:
        return 1.2
    if "nations league" in t:
        return 1.15
    if "friendly" in t:
        return 0.85
    return 1.0


def goal_diff_multiplier(home_goals: int, away_goals: int) -> float:
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11 + diff) / 8.0


def compute_elo_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id, date, home_team, away_team, home_score, away_score, "
            "tournament, neutral FROM matches ORDER BY date ASC, id ASC",
            conn,
        )

    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    history_rows = []

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh, ra = ratings[h], ratings[a]
        home_adv = 0.0 if row.neutral else ELO_HOME_ADV
        eh = expected_score(rh + home_adv, ra)
        ea = 1.0 - eh
        sh = actual_score(row.home_score, row.away_score, "home")
        sa = 1.0 - sh
        k = ELO_K * tournament_multiplier(row.tournament) * goal_diff_multiplier(
            row.home_score, row.away_score
        )
        new_rh = rh + k * (sh - eh)
        new_ra = ra + k * (sa - ea)

        history_rows.append((h, row.date, new_rh))
        history_rows.append((a, row.date, new_ra))
        ratings[h] = new_rh
        ratings[a] = new_ra

    hist = pd.DataFrame(history_rows, columns=["team", "date", "rating"])
    return hist, dict(ratings)


def store_elo(hist: pd.DataFrame):
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM elo")
        hist.to_sql("elo", conn, if_exists="append", index=False)
        conn.commit()


def main():
    hist, final = compute_elo_history()
    store_elo(hist)
    print(f"Stored {len(hist)} ELO rows for {len(final)} teams.")
    top = sorted(final.items(), key=lambda kv: -kv[1])[:15]
    print("Top 15 by final ELO:")
    for t, r in top:
        print(f"  {t:30s} {r:7.1f}")


if __name__ == "__main__":
    main()
