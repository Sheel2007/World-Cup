"""Render a knockout bracket as a tree-style matplotlib figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


ROUND_ORDER = ["Round of 32", "Round of 16", "Quarter-finals",
               "Semi-finals", "Final"]


def _draw_match(ax, x, y, w, h, team_a, team_b, winner, p_winner, match_no=None):
    """Draw a single two-row match box at (x, y) bottom-left."""
    # outer box
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                   edgecolor="#444", linewidth=0.8))
    # midline
    ax.plot([x, x + w], [y + h / 2, y + h / 2],
            color="#888", linewidth=0.5)

    rows = [
        (team_b, y, team_b == winner),       # bottom row
        (team_a, y + h / 2, team_a == winner),  # top row
    ]
    for name, ry, is_winner in rows:
        if is_winner:
            ax.add_patch(patches.Rectangle(
                (x, ry), w, h / 2, facecolor="#d4edda",
                edgecolor="none", alpha=0.85, zorder=0))
        weight = "bold" if is_winner else "normal"
        ax.text(x + 0.05 * w, ry + h / 4, name,
                fontsize=7.5, va="center", ha="left", fontweight=weight)

    if winner == team_a:
        py = y + 3 * h / 4
    else:
        py = y + h / 4
    ax.text(x + 0.97 * w, py, f"{p_winner * 100:.0f}%",
            fontsize=6.5, va="center", ha="right", color="#155724")
    if match_no is not None:
        ax.text(x + 0.03 * w, y + h - 0.02, f"M{match_no}",
                fontsize=5.5, va="top", ha="left", color="#6b7280")


def _draw_connector(ax, x0, y0, x1, y1, x_mid=None):
    """Draw stepped connector from (x0, y0) to (x1, y1)."""
    if x_mid is None:
        x_mid = (x0 + x1) / 2
    ax.plot([x0, x_mid, x_mid, x1], [y0, y0, y1, y1],
            color="#999", linewidth=0.7)


def render_bracket(prediction: dict, out_path: Path, title_suffix: str = ""):
    rounds = prediction["rounds"]
    if not rounds:
        return

    # Filter and order rounds present
    rounds_by_name = {r["round"]: r for r in rounds}
    present = [r for r in ROUND_ORDER if r in rounds_by_name]

    n_first = len(rounds_by_name[present[0]]["matches"])
    fig_h = max(8.0, 0.55 * 2 * n_first)
    fig_w = 4.2 * len(present)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    col_w = fig_w / (len(present) + 0.2)
    box_w = col_w * 0.85
    margin_x = col_w * 0.075

    # For each round compute per-match (cx, cy) center coords.
    match_centers: dict[tuple[int, int], tuple[float, float]] = {}

    for ci, rname in enumerate(present):
        matches = rounds_by_name[rname]["matches"]
        n = len(matches)
        # Spacing for this column: bracket halves the count each round → boxes double in spacing
        slot = fig_h / n
        box_h = min(slot * 0.78, 1.1)
        for mi, m in enumerate(matches):
            # Center this match between its two predecessors when possible
            if ci == 0:
                cy = fig_h - (mi + 0.5) * slot
            else:
                a = match_centers[(ci - 1, mi * 2)][1]
                b = match_centers[(ci - 1, mi * 2 + 1)][1]
                cy = (a + b) / 2
            x = ci * col_w + margin_x
            y = cy - box_h / 2
            _draw_match(ax, x, y, box_w, box_h,
                        m["team_a"], m["team_b"], m["winner"],
                        float(m["p_winner"]),
                        match_no=m.get("match_no"))
            match_centers[(ci, mi)] = (x + box_w, cy)

    # Draw connectors round k → round k+1
    for ci in range(len(present) - 1):
        next_matches = rounds_by_name[present[ci + 1]]["matches"]
        for mi, m in enumerate(next_matches):
            x_next = (ci + 1) * col_w + margin_x
            y_next = match_centers[(ci + 1, mi)][1]
            x0a, y0a = match_centers[(ci, mi * 2)]
            x0b, y0b = match_centers[(ci, mi * 2 + 1)]
            mid_x = (x0a + x_next) / 2
            _draw_connector(ax, x0a, y0a, x_next, y_next, x_mid=mid_x)
            _draw_connector(ax, x0b, y0b, x_next, y_next, x_mid=mid_x)

    # Round headers
    for ci, rname in enumerate(present):
        ax.text(ci * col_w + margin_x + box_w / 2, fig_h - 0.15,
                rname, fontsize=11, fontweight="bold", ha="center", va="top",
                color="#222")

    # Champion banner
    champ = prediction.get("champion", "?")
    fig.suptitle(f"2026 World Cup — Predicted Champion: {champ}{title_suffix}",
                 fontsize=14, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bracket → {out_path}")


def render_from_json(json_path: Path, out_path: Path, title_suffix: str = ""):
    with open(json_path) as f:
        prediction = json.load(f)
    render_bracket(prediction, out_path, title_suffix=title_suffix)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to prediction JSON")
    p.add_argument("--output", required=True, help="path to output PNG")
    p.add_argument("--suffix", default="", help="title suffix")
    args = p.parse_args()
    render_from_json(Path(args.input), Path(args.output), args.suffix)


if __name__ == "__main__":
    main()
