"""Per-match feature engineering.

Builds 14-dim feature vector + 3-class label per historical match.
Also exposes build_feature_for_pair() for live prediction of unseen matchups.
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MIN_TRAIN_YEAR
from data.db import get_conn


FEATURE_NAMES = [
    "elo_diff",
    "home_form5",
    "away_form5",
    "home_gf10",
    "home_ga10",
    "away_gf10",
    "away_ga10",
    "h2h_home_score",
    "home_rest_days",
    "away_rest_days",
    "neutral",
    "home_winrate20",
    "away_winrate20",
    "tournament_importance",
]


def tournament_importance(tournament: str) -> float:
    t = (tournament or "").lower()
    if "world cup" in t and "qualif" not in t:
        return 1.0
    if "uefa euro" in t or "copa am" in t or "africa cup" in t or "asian cup" in t:
        return 0.85
    if "qualif" in t:
        return 0.6
    if "nations league" in t:
        return 0.55
    if "friendly" in t:
        return 0.2
    return 0.4


def label_from_score(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 2  # home win
    if home_score < away_score:
        return 0  # away win
    return 1      # draw


def _form_value(my: int, opp: int) -> float:
    if my > opp:
        return 1.0
    if my < opp:
        return 0.0
    return 0.5


class FeatureState:
    """Rolling per-team state, updated chronologically as matches are processed."""

    def __init__(self):
        self.last_match_date: dict[str, str] = {}
        self.form: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self.gf10: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.ga10: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.results20: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=5))
        self.elo: dict[str, float] = {}

    def update_elo(self, elo_lookup: dict):
        self.elo = elo_lookup

    def features_for(self, home: str, away: str, date: str,
                     neutral: int, tournament: str) -> np.ndarray | None:
        # Skip until both teams have enough history.
        if (len(self.form[home]) < 3 or len(self.form[away]) < 3 or
                len(self.gf10[home]) < 5 or len(self.gf10[away]) < 5):
            return None

        elo_h = self.elo.get(home, 1500.0)
        elo_a = self.elo.get(away, 1500.0)
        elo_diff = (elo_h - elo_a) + (0 if neutral else 65.0)

        home_form5 = float(np.mean(self.form[home]))
        away_form5 = float(np.mean(self.form[away]))
        home_gf10 = float(np.mean(self.gf10[home]))
        home_ga10 = float(np.mean(self.ga10[home]))
        away_gf10 = float(np.mean(self.gf10[away]))
        away_ga10 = float(np.mean(self.ga10[away]))

        h2h_key = tuple(sorted([home, away]))
        h2h_records = self.h2h[h2h_key]
        if h2h_records:
            # records stored as (home_perspective_team, score)
            h2h_home_score = float(np.mean([
                s if t == home else 1.0 - s for (t, s) in h2h_records
            ]))
        else:
            h2h_home_score = 0.5

        def rest_days(team):
            prev = self.last_match_date.get(team)
            if not prev:
                return 30.0
            d0 = pd.Timestamp(prev)
            d1 = pd.Timestamp(date)
            return float(min((d1 - d0).days, 60))

        home_rest = rest_days(home)
        away_rest = rest_days(away)

        home_wr20 = float(np.mean([1.0 if r == 1 else 0.0 for r in self.results20[home]])) \
            if self.results20[home] else 0.5
        away_wr20 = float(np.mean([1.0 if r == 1 else 0.0 for r in self.results20[away]])) \
            if self.results20[away] else 0.5

        importance = tournament_importance(tournament)

        return np.array([
            elo_diff,
            home_form5,
            away_form5,
            home_gf10,
            home_ga10,
            away_gf10,
            away_ga10,
            h2h_home_score,
            home_rest,
            away_rest,
            float(neutral),
            home_wr20,
            away_wr20,
            importance,
        ], dtype=np.float64)

    def post_match_update(self, home: str, away: str, date: str,
                          hs: int, as_: int):
        # form
        self.form[home].append(_form_value(hs, as_))
        self.form[away].append(_form_value(as_, hs))
        # goals
        self.gf10[home].append(hs)
        self.ga10[home].append(as_)
        self.gf10[away].append(as_)
        self.ga10[away].append(hs)
        # win indicator for overall winrate
        self.results20[home].append(1 if hs > as_ else 0)
        self.results20[away].append(1 if as_ > hs else 0)
        # h2h record: store (winner-perspective-team, score_from_home_team_perspective)
        h2h_key = tuple(sorted([home, away]))
        self.h2h[h2h_key].append((home, _form_value(hs, as_)))
        # last match date
        self.last_match_date[home] = date
        self.last_match_date[away] = date


def _load_elo_lookup() -> dict[tuple, float]:
    """Map (team, date) -> rating as of that date (rating *after* that match)."""
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT team, date, rating FROM elo ORDER BY date", conn)
    return df


def _elo_before(elo_df: pd.DataFrame, team: str, date: str) -> float:
    sub = elo_df[(elo_df["team"] == team) & (elo_df["date"] < date)]
    if sub.empty:
        return 1500.0
    return float(sub.iloc[-1]["rating"])


def build_training_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Walk matches chronologically, emit features + labels for every match
    where both teams already have enough history.
    """
    with get_conn() as conn:
        matches = pd.read_sql_query(
            "SELECT id, date, home_team, away_team, home_score, away_score, "
            "tournament, neutral FROM matches ORDER BY date ASC, id ASC",
            conn,
        )
        elo_df = pd.read_sql_query("SELECT team, date, rating FROM elo ORDER BY date", conn)

    # build per-team sorted arrays for fast lookup of rating-as-of-date
    elo_by_team = {t: g.reset_index(drop=True) for t, g in elo_df.groupby("team")}

    state = FeatureState()
    rows = []
    for m in matches.itertuples(index=False):
        # lookup ELO BEFORE this match
        def rating_before(team):
            g = elo_by_team.get(team)
            if g is None:
                return 1500.0
            idx = g["date"].searchsorted(m.date, side="left")
            if idx == 0:
                return 1500.0
            return float(g.iloc[idx - 1]["rating"])

        state.elo = {m.home_team: rating_before(m.home_team),
                     m.away_team: rating_before(m.away_team)}

        feats = state.features_for(m.home_team, m.away_team, m.date,
                                   m.neutral, m.tournament)
        if feats is not None:
            label = label_from_score(m.home_score, m.away_score)
            rows.append((m.id, m.date, m.home_team, m.away_team, feats, label))

        state.post_match_update(m.home_team, m.away_team, m.date,
                                m.home_score, m.away_score)

    if not rows:
        raise RuntimeError("No feature rows produced.")

    ids = [r[0] for r in rows]
    dates = [r[1] for r in rows]
    homes = [r[2] for r in rows]
    aways = [r[3] for r in rows]
    X = np.stack([r[4] for r in rows])
    y = np.array([r[5] for r in rows], dtype=np.int64)

    meta = pd.DataFrame({"match_id": ids, "date": dates,
                         "home_team": homes, "away_team": aways})
    return meta, X, y


def build_state_through(cutoff_date: str) -> tuple[FeatureState, dict]:
    """Replay history up to cutoff and return state + per-team ELO snapshot."""
    with get_conn() as conn:
        matches = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score, "
            "tournament, neutral FROM matches WHERE date <= ? "
            "ORDER BY date ASC, id ASC",
            conn,
            params=(cutoff_date,),
        )
        elo_df = pd.read_sql_query(
            "SELECT team, date, rating FROM elo WHERE date <= ? ORDER BY date",
            conn,
            params=(cutoff_date,),
        )

    state = FeatureState()
    for m in matches.itertuples(index=False):
        state.post_match_update(m.home_team, m.away_team, m.date,
                                m.home_score, m.away_score)

    final_elo = (elo_df.sort_values("date")
                 .drop_duplicates("team", keep="last")
                 .set_index("team")["rating"].to_dict())
    state.elo = final_elo
    return state, final_elo


def build_feature_for_pair(state: FeatureState, home: str, away: str,
                           date: str, neutral: int = 1,
                           tournament: str = "FIFA World Cup") -> np.ndarray:
    """Build a single feature row for a hypothetical matchup at given date."""
    feats = state.features_for(home, away, date, neutral, tournament)
    if feats is None:
        # fallback: skip min-history guard for tournament prediction
        elo_h = state.elo.get(home, 1500.0)
        elo_a = state.elo.get(away, 1500.0)
        elo_diff = (elo_h - elo_a) + (0 if neutral else 65.0)
        home_form5 = float(np.mean(state.form[home])) if state.form[home] else 0.5
        away_form5 = float(np.mean(state.form[away])) if state.form[away] else 0.5
        home_gf10 = float(np.mean(state.gf10[home])) if state.gf10[home] else 1.2
        home_ga10 = float(np.mean(state.ga10[home])) if state.ga10[home] else 1.2
        away_gf10 = float(np.mean(state.gf10[away])) if state.gf10[away] else 1.2
        away_ga10 = float(np.mean(state.ga10[away])) if state.ga10[away] else 1.2
        h2h_key = tuple(sorted([home, away]))
        h2h_records = state.h2h[h2h_key]
        h2h_home_score = (float(np.mean([s if t == home else 1.0 - s
                                          for (t, s) in h2h_records]))
                          if h2h_records else 0.5)
        home_wr20 = (float(np.mean([1.0 if r == 1 else 0.0 for r in state.results20[home]]))
                     if state.results20[home] else 0.5)
        away_wr20 = (float(np.mean([1.0 if r == 1 else 0.0 for r in state.results20[away]]))
                     if state.results20[away] else 0.5)
        feats = np.array([
            elo_diff, home_form5, away_form5,
            home_gf10, home_ga10, away_gf10, away_ga10,
            h2h_home_score, 7.0, 7.0,
            float(neutral), home_wr20, away_wr20,
            tournament_importance(tournament),
        ], dtype=np.float64)
    return feats


def persist_features(meta: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM features")
        for i, row in meta.iterrows():
            cur.execute(
                "INSERT OR REPLACE INTO features (match_id, feature_json, label) "
                "VALUES (?, ?, ?)",
                (int(row["match_id"]), json.dumps(X[i].tolist()), int(y[i])),
            )
        conn.commit()


def main():
    meta, X, y = build_training_features()
    print(f"Built {len(meta)} feature rows, shape={X.shape}")
    print(f"Label distribution: away_win={int((y==0).sum())} "
          f"draw={int((y==1).sum())} home_win={int((y==2).sum())}")
    persist_features(meta, X, y)
    print(f"Persisted to features table.")


if __name__ == "__main__":
    main()
