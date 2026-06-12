"""Group stage simulation — round-robin in each of the 12 groups."""
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import WC_2026_GROUPS, WC_2026_HOSTS
from features.builder import FeatureState, build_feature_for_pair


PredictFn = Callable[[np.ndarray], np.ndarray]  # (n, 14) -> (n, 3) [away, draw, home]


def predict_match(predict_fn: PredictFn, state: FeatureState,
                  home: str, away: str, date: str,
                  neutral: int = 1, tournament: str = "FIFA World Cup") -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) for a single matchup."""
    feats = build_feature_for_pair(state, home, away, date, neutral, tournament)
    probs = predict_fn(feats.reshape(1, -1))[0]
    # data_loader label order: 0=away win, 1=draw, 2=home win
    return float(probs[2]), float(probs[1]), float(probs[0])


def simulate_group(group_name: str, teams: list[str], predict_fn: PredictFn,
                   state: FeatureState, base_date: str = "2026-06-15") -> dict:
    points = {t: 0.0 for t in teams}
    gf = {t: 0.0 for t in teams}
    ga = {t: 0.0 for t in teams}
    match_records = []

    for ta, tb in combinations(teams, 2):
        # USA/Canada/Mexico home if playing in group; else neutral
        if ta in WC_2026_HOSTS and tb not in WC_2026_HOSTS:
            home, away, neutral = ta, tb, 0
        elif tb in WC_2026_HOSTS and ta not in WC_2026_HOSTS:
            home, away, neutral = tb, ta, 0
        else:
            home, away, neutral = ta, tb, 1

        pH, pD, pA = predict_match(predict_fn, state, home, away,
                                   base_date, neutral=neutral)
        # expected points (3 win, 1 draw)
        points[home] += 3 * pH + 1 * pD
        points[away] += 3 * pA + 1 * pD
        # expected goals — use form averages as proxy
        gf_h = float(np.mean(state.gf10[home])) if state.gf10[home] else 1.2
        gf_a = float(np.mean(state.gf10[away])) if state.gf10[away] else 1.2
        gf[home] += gf_h * (pH + 0.5 * pD) + 0.5 * gf_h * pA
        gf[away] += gf_a * (pA + 0.5 * pD) + 0.5 * gf_a * pH
        ga[home] += gf_a * (pA + 0.5 * pD) + 0.5 * gf_a * pH
        ga[away] += gf_h * (pH + 0.5 * pD) + 0.5 * gf_h * pA
        match_records.append({
            "home": home, "away": away, "neutral": neutral,
            "p_home": pH, "p_draw": pD, "p_away": pA,
        })

    standings = sorted(
        teams,
        key=lambda t: (-points[t], -(gf[t] - ga[t]), -gf[t]),
    )
    return {
        "group": group_name,
        "teams": teams,
        "standings": standings,
        "points": points,
        "gf": gf,
        "ga": ga,
        "matches": match_records,
    }


def simulate_all_groups(predict_fn: PredictFn, state: FeatureState) -> dict:
    out = {}
    for g, teams in WC_2026_GROUPS.items():
        out[g] = simulate_group(g, teams, predict_fn, state)
    return out


def qualified_teams(group_results: dict) -> list[tuple[str, str, int]]:
    """Return list of (team, group, seed) for 32-team knockout.
    Top 2 from each group (24) + 8 best third-place teams.
    """
    top_two = []
    third_place = []
    for g, res in group_results.items():
        s = res["standings"]
        top_two.append((s[0], g, 1))
        top_two.append((s[1], g, 2))
        third_place.append((s[2], g, 3, res["points"][s[2]],
                            res["gf"][s[2]] - res["ga"][s[2]],
                            res["gf"][s[2]]))

    # Sort third-placers, pick best 8
    third_place.sort(key=lambda t: (-t[3], -t[4], -t[5]))
    best_third = [(t[0], t[1], t[2]) for t in third_place[:8]]

    return top_two + best_third
