"""Full tournament runner using the actual FIFA 2026 knockout bracket.

After the group stage, the Round of 32 has 16 fixed match slots (matches 73–88
in FIFA's numbering). Eight of those slots are wild-cards that take the
third-placed team from one of five specific groups; the assignment depends on
which 8 of 12 third-placed teams actually qualify (FIFA's regulations enumerate
all 495 combinations). We solve the assignment via bipartite matching against
the slot's allowed source-group set.

From Round of 16 onwards the bracket is a fixed match tree (89–96 → 97–100 →
101–102 → 104).
"""
import sys
from itertools import permutations
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import WC_2026_R32, WC_2026_R16, WC_2026_QF, WC_2026_SF, WC_2026_FINAL
from features.builder import FeatureState
from bracket.group_stage import simulate_all_groups, qualified_teams
from bracket.qaoa_optimizer import simulate_round


def _parse_slot(slot: str) -> dict:
    """Return {'kind': '1'|'2'|'3', 'group': 'A'} or
              {'kind': '3wild', 'allowed': ['A','B','C','D','F']}."""
    if slot.startswith("3:"):
        return {"kind": "3wild", "allowed": list(slot[2:])}
    return {"kind": slot[0], "group": slot[1]}


def _assign_thirds_to_slots(qualifying_thirds: dict, r32: list) -> dict:
    """Bipartite-match qualifying third-place groups to their R32 wild-card slots.

    qualifying_thirds: {group_letter: team_name}  (exactly 8 entries)
    r32: list of (match_no, slot_top, slot_bottom) tuples.

    Returns {match_no: group_letter} for the 8 wild-card slots.
    """
    wild_slots = []
    for match_no, top, bottom in r32:
        for slot in (top, bottom):
            parsed = _parse_slot(slot)
            if parsed["kind"] == "3wild":
                wild_slots.append((match_no, parsed["allowed"]))

    qualifying_groups = list(qualifying_thirds.keys())
    if len(wild_slots) != len(qualifying_groups):
        raise ValueError(f"Wild slots ({len(wild_slots)}) != "
                         f"qualifying thirds ({len(qualifying_groups)})")

    # Greedy by most-constrained slot first (allowed ∩ qualifying smallest)
    remaining = set(qualifying_groups)
    slots_sorted = sorted(
        wild_slots,
        key=lambda s: len(set(s[1]) & set(qualifying_groups))
    )
    assignment = {}
    for match_no, allowed in slots_sorted:
        candidates = [g for g in allowed if g in remaining]
        if not candidates:
            # back-track via exhaustive search (rare; runs at most 8!)
            return _exhaustive_third_assignment(wild_slots, qualifying_groups)
        pick = candidates[0]
        assignment[match_no] = pick
        remaining.discard(pick)
    return assignment


def _exhaustive_third_assignment(wild_slots, qualifying_groups):
    """Fallback: try all 8! permutations until one fits every slot's allowed set."""
    for perm in permutations(qualifying_groups):
        ok = True
        assignment = {}
        for (match_no, allowed), grp in zip(wild_slots, perm):
            if grp not in allowed:
                ok = False
                break
            assignment[match_no] = grp
        if ok:
            return assignment
    raise RuntimeError("No feasible third-place assignment found.")


def _resolve_r32(group_results: dict, qualified: list) -> list[tuple[int, str, str]]:
    """Return [(match_no, team_top, team_bottom), …] for all 16 R32 matches."""
    standings = {g: res["standings"] for g, res in group_results.items()}
    qualifying_thirds = {g: t for (t, g, seed) in qualified if seed == 3}
    assignment = _assign_thirds_to_slots(qualifying_thirds, WC_2026_R32)

    def team_of(slot: str, match_no: int) -> str:
        p = _parse_slot(slot)
        if p["kind"] == "1":
            return standings[p["group"]][0]
        if p["kind"] == "2":
            return standings[p["group"]][1]
        if p["kind"] == "3wild":
            g = assignment[match_no]
            return qualifying_thirds[g]
        raise ValueError(slot)

    r32_resolved = []
    for match_no, top, bottom in WC_2026_R32:
        r32_resolved.append((match_no, team_of(top, match_no),
                             team_of(bottom, match_no)))
    return r32_resolved


def _next_round_adjacency(round_matches: list, next_round_ties: list) -> list[tuple[int, int]]:
    """For QAOA coupling: map indices of matches that are paired in next round."""
    idx_by_no = {m[0]: i for i, m in enumerate(round_matches)}
    pairs = []
    for _, a, b in next_round_ties:
        if a in idx_by_no and b in idx_by_no:
            pairs.append((idx_by_no[a], idx_by_no[b]))
    return pairs


def run_tournament(predict_fn, state: FeatureState,
                   group_date: str = "2026-06-15",
                   ko_date: str = "2026-07-01") -> dict:
    print("\n=== GROUP STAGE ===")
    group_results = simulate_all_groups(predict_fn, state)
    for g, res in group_results.items():
        print(f"  Group {g}: " + ", ".join(
            f"{t}({res['points'][t]:.1f})" for t in res["standings"]
        ))

    qualified = qualified_teams(group_results)
    qualifying_thirds = [(t, g) for (t, g, seed) in qualified if seed == 3]
    print(f"\nBest 8 third-place teams advancing:")
    for t, g in qualifying_thirds:
        print(f"  Group {g} 3rd: {t}")

    # --- Round of 32 (FIFA fixed bracket + bipartite 3rd-place assignment) ---
    r32_resolved = _resolve_r32(group_results, qualified)
    print(f"\n=== ROUND OF 32 ===")
    for mno, ta, tb in r32_resolved:
        print(f"  M{mno}: {ta} vs {tb}")
    pairs_r32 = [(ta, tb) for (_, ta, tb) in r32_resolved]
    r32_result = simulate_round(
        "Round of 32", pairs_r32, predict_fn, state, ko_date,
        next_round_pairs=_next_round_adjacency(r32_resolved, WC_2026_R16),
    )
    print(f"  → winners: {r32_result['winners']}")

    winners_by_match = {mno: r32_result["winners"][i]
                        for i, (mno, _, _) in enumerate(r32_resolved)}

    # --- Helper: run any fixed-tree round given the FIFA tie list ---
    def run_fixed_round(name: str, ties: list, next_ties: list | None) -> dict:
        print(f"\n=== {name} ===")
        round_matches = []
        for mno, a, b in ties:
            ta = winners_by_match[a]
            tb = winners_by_match[b]
            print(f"  M{mno}: {ta} vs {tb}")
            round_matches.append((mno, ta, tb))
        pairs = [(ta, tb) for (_, ta, tb) in round_matches]
        adj = (_next_round_adjacency(round_matches, next_ties)
               if next_ties else None)
        res = simulate_round(name, pairs, predict_fn, state, ko_date,
                             next_round_pairs=adj)
        for i, (mno, _, _) in enumerate(round_matches):
            winners_by_match[mno] = res["winners"][i]
        print(f"  → winners: {res['winners']}")
        return res

    r16_result = run_fixed_round("Round of 16", WC_2026_R16, WC_2026_QF)
    qf_result = run_fixed_round("Quarter-finals", WC_2026_QF, WC_2026_SF)
    sf_result = run_fixed_round("Semi-finals", WC_2026_SF, [WC_2026_FINAL])
    final_result = run_fixed_round("Final", [WC_2026_FINAL], None)

    champion = final_result["winners"][0]
    print(f"\n*** PREDICTED CHAMPION: {champion} ***")

    rounds = [r32_result, r16_result, qf_result, sf_result, final_result]
    _tag_with_match_nos(rounds, r32_resolved)
    rounds = _reorder_for_bracket_tree(rounds)

    return {
        "groups": group_results,
        "qualified": qualified,
        "rounds": rounds,
        "champion": champion,
    }


def _tag_with_match_nos(rounds: list, r32_resolved: list):
    """Attach FIFA match number to each match dict."""
    # R32: order is r32_resolved order (73…88)
    for i, (mno, _, _) in enumerate(r32_resolved):
        rounds[0]["matches"][i]["match_no"] = mno
    # R16 / QF / SF / Final from config tie lists
    for round_idx, ties in [(1, WC_2026_R16), (2, WC_2026_QF),
                            (3, WC_2026_SF), (4, [WC_2026_FINAL])]:
        for i, (mno, _, _) in enumerate(ties):
            rounds[round_idx]["matches"][i]["match_no"] = mno


def _reorder_for_bracket_tree(rounds: list) -> list:
    """Reorder each round so adjacent pairs (2i, 2i+1) feed the next round's
    match i. The Final fixes the leaf order; walk backwards.
    """
    # Build a map from match_no → its match dict (per round)
    per_round_by_mno = []
    for r in rounds:
        per_round_by_mno.append({m["match_no"]: m for m in r["matches"]})

    # Start from final and walk back through the tie structure.
    final_mno = WC_2026_FINAL[0]
    desired_order = [[final_mno]]

    sf_ties_by_mno = {mno: (a, b) for (mno, a, b) in WC_2026_SF}
    qf_ties_by_mno = {mno: (a, b) for (mno, a, b) in WC_2026_QF}
    r16_ties_by_mno = {mno: (a, b) for (mno, a, b) in WC_2026_R16}
    final_a, final_b = WC_2026_FINAL[1], WC_2026_FINAL[2]
    desired_order.insert(0, [final_a, final_b])

    # QF order
    qf_order = []
    for sf_mno in desired_order[0]:
        a, b = sf_ties_by_mno[sf_mno]
        qf_order.extend([a, b])
    desired_order.insert(0, qf_order)

    # R16 order
    r16_order = []
    for qf_mno in qf_order:
        a, b = qf_ties_by_mno[qf_mno]
        r16_order.extend([a, b])
    desired_order.insert(0, r16_order)

    # R32 order
    r32_order = []
    for r16_mno in r16_order:
        a, b = r16_ties_by_mno[r16_mno]
        r32_order.extend([a, b])
    desired_order.insert(0, r32_order)

    # Now apply: for each round, reorder its matches per desired_order
    new_rounds = []
    for round_idx, order in enumerate(desired_order):
        new_matches = [per_round_by_mno[round_idx][mno] for mno in order]
        new_rounds.append({
            **rounds[round_idx],
            "matches": new_matches,
            "winners": [m["winner"] for m in new_matches],
        })
    return new_rounds
