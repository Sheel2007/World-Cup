"""QAOA-based knockout round optimizer.

For each round, encode the match outcomes as a QUBO and solve with QAOA:
  - Variables: x_i in {0,1} per match. x_i = 1 means home/team-A wins.
  - Linear coefficient: -log(p_winner_i) — reward picking the more probable winner.
  - Quadratic regularizer: small penalty for sibling matches (adjacent bracket
    pairings) where both picks are toss-ups, which encourages stable paths.

QAOA runs with p=QAOA_REPS layers on Aer statevector simulator using COBYLA.
"""
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import QAOA_REPS
from features.builder import FeatureState, build_feature_for_pair


PredictFn = Callable[[np.ndarray], np.ndarray]


def _safe_log(p: float, eps: float = 1e-6) -> float:
    return float(np.log(max(p, eps)))


def _knockout_match_probs(predict_fn: PredictFn, state: FeatureState,
                          pairs: list[tuple[str, str]],
                          date: str) -> list[tuple[float, float]]:
    """For each pair, return (p_A_wins_outright, p_B_wins_outright) renormalised
    over the draw — knockout has no draws (penalty shootout, 50/50 from draw mass).
    """
    out = []
    for a, b in pairs:
        feats = build_feature_for_pair(state, a, b, date, neutral=1,
                                       tournament="FIFA World Cup")
        probs = predict_fn(feats.reshape(1, -1))[0]
        # labels: 0=away/B wins, 1=draw, 2=home/A wins
        p_a = probs[2] + 0.5 * probs[1]
        p_b = probs[0] + 0.5 * probs[1]
        s = p_a + p_b
        out.append((float(p_a / s), float(p_b / s)))
    return out


def _build_qubo(pair_probs: list[tuple[float, float]],
                next_round_pairs: list[tuple[int, int]] | None = None,
                lambda_reg: float = 0.05):
    """Build QuadraticProgram for one round.
    x_i = 1 → team A (first) wins; x_i = 0 → team B wins.
    Minimise -sum_i [log p_a_i * x_i + log p_b_i * (1-x_i)]
             + lambda * sum_{paired (i, j)} (x_i - 0.5)(x_j - 0.5)  (toss-up coupling)

    `next_round_pairs` lists which match indices' winners face each other in the
    next round (real FIFA bracket adjacency, not just consecutive indices). If
    omitted, falls back to (0,1), (2,3), …
    """
    from qiskit_optimization import QuadraticProgram

    qp = QuadraticProgram(name="knockout_round")
    n = len(pair_probs)
    for i in range(n):
        qp.binary_var(f"x{i}")

    linear = {}
    constant = 0.0
    for i, (pa, pb) in enumerate(pair_probs):
        la, lb = _safe_log(pa), _safe_log(pb)
        linear[f"x{i}"] = lb - la
        constant += -lb

    if next_round_pairs is None:
        next_round_pairs = [(i, i + 1) for i in range(0, n - 1, 2)]

    quadratic = {}
    for i, j in next_round_pairs:
        # toss-up coupling between matches whose winners actually meet next round
        # lambda * (x_i - 0.5)(x_j - 0.5)
        key = (f"x{min(i,j)}", f"x{max(i,j)}")
        quadratic[key] = quadratic.get(key, 0.0) + lambda_reg
        linear[f"x{i}"] = linear.get(f"x{i}", 0.0) - 0.5 * lambda_reg
        linear[f"x{j}"] = linear.get(f"x{j}", 0.0) - 0.5 * lambda_reg
        constant += 0.25 * lambda_reg

    qp.minimize(constant=constant, linear=linear, quadratic=quadratic)
    return qp


def _solve_with_qaoa(qp):
    from qiskit_aer.primitives import Sampler as AerSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    sampler = AerSampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=80), reps=QAOA_REPS)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    return result


def simulate_round(round_name: str, pairs: list[tuple[str, str]],
                   predict_fn: PredictFn, state: FeatureState,
                   date: str,
                   next_round_pairs: list[tuple[int, int]] | None = None) -> dict:
    pair_probs = _knockout_match_probs(predict_fn, state, pairs, date)
    qp = _build_qubo(pair_probs, next_round_pairs=next_round_pairs)
    result = _solve_with_qaoa(qp)
    # x_i=1 → team A wins; x_i=0 → team B wins
    bits = [int(round(v)) for v in result.x]
    winners = [pairs[i][0] if bits[i] == 1 else pairs[i][1]
               for i in range(len(pairs))]
    match_log = []
    for i, (a, b) in enumerate(pairs):
        pa, pb = pair_probs[i]
        match_log.append({
            "round": round_name, "team_a": a, "team_b": b,
            "p_a": pa, "p_b": pb,
            "winner": winners[i],
            "p_winner": pa if bits[i] == 1 else pb,
        })
    return {
        "round": round_name,
        "winners": winners,
        "matches": match_log,
        "qaoa_objective": float(result.fval),
    }


def make_pairs(teams: list[str]) -> list[tuple[str, str]]:
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
