"""Shared loader: features → numpy arrays split by date cutoff."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAIN_CUTOFF, TEST_CUTOFF
from data.db import get_conn


def load_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """SELECT f.match_id, f.feature_json, f.label, m.date,
                      m.home_team, m.away_team
               FROM features f JOIN matches m ON m.id = f.match_id
               ORDER BY m.date ASC""",
            conn,
        )
    X = np.stack([np.array(json.loads(s)) for s in df["feature_json"]])
    y = df["label"].to_numpy().astype(np.int64)
    meta = df[["match_id", "date", "home_team", "away_team"]].reset_index(drop=True)
    return meta, X, y


def split_by_date(meta: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    train_mask = meta["date"] < TRAIN_CUTOFF
    test_mask = (meta["date"] >= TRAIN_CUTOFF) & (meta["date"] <= TEST_CUTOFF)
    return (X[train_mask], y[train_mask], meta[train_mask].reset_index(drop=True),
            X[test_mask], y[test_mask], meta[test_mask].reset_index(drop=True))
