"""End-to-end 2026 World Cup prediction.

Pipeline: load chosen model → replay history → simulate groups → QAOA knockout.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR
from features.builder import build_state_through
from models import qsvm as qsvm_mod
from models import baseline as xgb_mod
from bracket.runner import run_tournament
from bracket.visualize import render_bracket


def get_predictor(name: str):
    if name == "qsvm":
        return qsvm_mod.predict_proba
    if name == "xgb":
        return xgb_mod.predict_proba
    raise ValueError(f"Unknown model: {name}")




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qsvm", "xgb"], default="xgb",
                   help="Which match predictor to use")
    p.add_argument("--cutoff", default="2026-06-01",
                   help="Use match history up to this date when building features")
    p.add_argument("--out", default=str(OUTPUT_DIR / "prediction.json"))
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predict_fn = get_predictor(args.model)
    state, _ = build_state_through(args.cutoff)
    print(f"State replayed through {args.cutoff} — {len(state.elo)} teams known")

    result = run_tournament(predict_fn, state)

    # Serialise group stage results (drop non-JSON-friendly fields).
    groups_out = {}
    for g, res in result["groups"].items():
        groups_out[g] = {
            "group": g,
            "teams": res["teams"],
            "standings": res["standings"],
            "points": {t: round(float(res["points"][t]), 3) for t in res["teams"]},
            "gf": {t: round(float(res["gf"][t]), 3) for t in res["teams"]},
            "ga": {t: round(float(res["ga"][t]), 3) for t in res["teams"]},
            "matches": res["matches"],
        }

    out_path = Path(args.out)
    with out_path.open("w") as f:
        json.dump({
            "model": args.model,
            "champion": result["champion"],
            "rounds": result["rounds"],
            "qualified": [list(q) for q in result["qualified"]],
            "groups": groups_out,
        }, f, indent=2)
    print(f"Saved JSON → {out_path}")

    render_bracket(
        {"rounds": result["rounds"], "champion": result["champion"]},
        OUTPUT_DIR / f"bracket_{args.model}.png",
        title_suffix=f"  ({args.model.upper()})",
    )


if __name__ == "__main__":
    main()
