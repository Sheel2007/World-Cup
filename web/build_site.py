"""Build outputs/index.html — a single-file static site showing the full
predicted tournament (group stage, qualified third-place teams, knockout
bracket) for both XGBoost and QSVM. Open directly with file:// — no server.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR


XGB_JSON = OUTPUT_DIR / "prediction_xgb.json"
QSVM_JSON = OUTPUT_DIR / "prediction_qsvm.json"
OUTPUT_HTML = OUTPUT_DIR / "index.html"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quantum World Cup Predictor — 2026</title>
<style>
  :root {
    --bg: #f6f7fb;
    --panel: #ffffff;
    --ink: #1f2937;
    --muted: #64748b;
    --line: #e2e8f0;
    --accent: #1e40af;
    --winner: #d1fae5;
    --winner-edge: #047857;
    --third: #fef3c7;
    --third-edge: #b45309;
    --runner: #e0f2fe;
    --runner-edge: #0369a1;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, sans-serif;
    background: var(--bg);
    color: var(--ink);
    font-size: 14px;
  }
  header {
    background: linear-gradient(120deg, #1e3a8a 0%, #4c1d95 100%);
    color: white;
    padding: 24px 32px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }
  h1 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
  header .sub { font-size: 13px; opacity: 0.85; }
  .tabs {
    display: flex;
    gap: 8px;
    padding: 16px 32px 0 32px;
    background: var(--bg);
    border-bottom: 1px solid var(--line);
  }
  .tab {
    padding: 10px 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    cursor: pointer;
    font-weight: 600;
    color: var(--muted);
  }
  .tab.active { color: var(--accent); border-color: var(--accent); border-bottom: 2px solid var(--panel); margin-bottom: -1px; }
  main { padding: 24px 32px 64px 32px; }
  .champion-banner {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    border-left: 5px solid var(--accent);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
  }
  .champion-banner .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
  .champion-banner .champ { font-size: 28px; font-weight: 800; color: var(--accent); }
  .champion-banner .meta { font-size: 12px; color: var(--muted); }
  section { margin-bottom: 32px; }
  section h2 { font-size: 16px; margin: 0 0 12px 0; color: var(--ink); }
  section h2 .count { font-weight: 400; color: var(--muted); font-size: 13px; margin-left: 8px; }

  .groups-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
  }
  .group-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .group-card h3 { margin: 0 0 8px 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .group-card table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .group-card th, .group-card td { text-align: left; padding: 4px 6px; }
  .group-card th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; border-bottom: 1px solid var(--line); }
  .group-card td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .group-card tr.seed-1 td { background: var(--winner); }
  .group-card tr.seed-2 td { background: var(--runner); }
  .group-card tr.seed-3 td { background: var(--third); }
  .group-card tr.seed-1 td:first-child { border-left: 3px solid var(--winner-edge); }
  .group-card tr.seed-2 td:first-child { border-left: 3px solid var(--runner-edge); }
  .group-card tr.seed-3 td:first-child { border-left: 3px solid var(--third-edge); }
  .legend { display: flex; gap: 12px; font-size: 12px; color: var(--muted); margin: 8px 0 16px 0; flex-wrap: wrap; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .thirds-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 18px;
  }
  .thirds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
  .thirds-grid .chip {
    background: var(--third);
    border: 1px solid var(--third-edge);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
  }
  .thirds-grid .chip span.g { color: var(--third-edge); font-weight: 500; font-size: 11px; }

  .bracket {
    display: flex;
    gap: 18px;
    overflow-x: auto;
    padding-bottom: 8px;
  }
  .round-col {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    min-width: 220px;
  }
  .round-col h3 { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 12px 0; text-align: center; }
  .match {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin: 6px 0;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .match-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    font-size: 13px;
  }
  .match-row + .match-row { border-top: 1px solid var(--line); }
  .match-row.winner { background: var(--winner); font-weight: 700; }
  .match-row .prob { color: var(--winner-edge); font-variant-numeric: tabular-nums; font-size: 12px; }
  .match-label { font-size: 10px; color: var(--muted); padding: 2px 10px 0 10px; font-weight: 500; }

  .matches-detail {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--muted);
    max-height: 220px;
    overflow-y: auto;
  }
  .matches-detail summary { cursor: pointer; font-weight: 600; color: var(--ink); }
  .matches-detail table { width: 100%; border-collapse: collapse; margin-top: 6px; }
  .matches-detail td { padding: 2px 6px; }

  footer { padding: 24px 32px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); background: var(--panel); }
</style>
</head>
<body>
<header>
  <h1>Quantum World Cup Predictor — 2026</h1>
  <div class="sub">QSVM match probabilities · QAOA bracket optimisation · XGBoost classical baseline</div>
</header>

<div class="tabs">
  <div class="tab active" data-model="xgb">XGBoost + QAOA</div>
  <div class="tab" data-model="qsvm">QSVM + QAOA</div>
</div>

<main>
  <div class="champion-banner">
    <div>
      <div class="label">Predicted champion</div>
      <div class="champ" id="champ">—</div>
    </div>
    <div class="meta" id="meta">—</div>
  </div>

  <section>
    <h2>Group stage <span class="count" id="groups-count"></span></h2>
    <div class="legend">
      <span><i style="background: var(--winner); border: 1px solid var(--winner-edge)"></i>1st (auto-qualify)</span>
      <span><i style="background: var(--runner); border: 1px solid var(--runner-edge)"></i>2nd (auto-qualify)</span>
      <span><i style="background: var(--third); border: 1px solid var(--third-edge)"></i>3rd (8 best advance)</span>
    </div>
    <div class="groups-grid" id="groups"></div>
  </section>

  <section>
    <h2>Best third-place qualifiers <span class="count">Top 8 of 12 third-placed teams</span></h2>
    <div class="thirds-card"><div class="thirds-grid" id="thirds"></div></div>
  </section>

  <section>
    <h2>Knockout bracket <span class="count">R32 → R16 → QF → SF → Final</span></h2>
    <div class="bracket" id="bracket"></div>
  </section>
</main>

<footer>
  Built from <code>outputs/prediction_xgb.json</code> + <code>outputs/prediction_qsvm.json</code>.
  Group standings, qualifiers and knockout outcomes all driven by the selected model;
  knockout rounds additionally optimised with QAOA on a Qiskit Aer simulator.
</footer>

<script>
const DATA = __DATA_JSON__;

function fmt(n, d=1) { return Number(n).toFixed(d); }
function pct(p) { return Math.round(p * 100) + "%"; }

function renderGroups(model) {
  const groups = DATA[model].groups;
  const container = document.getElementById("groups");
  container.innerHTML = "";

  // Rebuild qualified-third lookup so we can highlight the 8 best third-placers
  // exactly as the seeding code does.
  const thirds = [];
  for (const g of Object.keys(groups).sort()) {
    const res = groups[g];
    const third = res.standings[2];
    thirds.push({
      team: third, group: g,
      pts: res.points[third], gd: res.gf[third] - res.ga[third], gf: res.gf[third],
    });
  }
  thirds.sort((a, b) => (b.pts - a.pts) || (b.gd - a.gd) || (b.gf - a.gf));
  const qualifiedThirds = new Set(thirds.slice(0, 8).map(t => `${t.group}:${t.team}`));

  for (const g of Object.keys(groups).sort()) {
    const res = groups[g];
    const card = document.createElement("div");
    card.className = "group-card";
    let rows = res.standings.map((team, idx) => {
      const pts = res.points[team], gf = res.gf[team], ga = res.ga[team];
      let cls = "";
      if (idx === 0) cls = "seed-1";
      else if (idx === 1) cls = "seed-2";
      else if (idx === 2 && qualifiedThirds.has(`${g}:${team}`)) cls = "seed-3";
      return `<tr class="${cls}">
        <td>${team}</td>
        <td class="num">${fmt(pts)}</td>
        <td class="num">${fmt(gf)}</td>
        <td class="num">${fmt(ga)}</td>
        <td class="num">${fmt(gf - ga, 1)}</td>
      </tr>`;
    }).join("");

    let matchRows = res.matches.map(m =>
      `<tr><td>${m.home}</td><td class="num">${pct(m.p_home)}</td><td class="num">D ${pct(m.p_draw)}</td><td class="num">${pct(m.p_away)}</td><td>${m.away}</td></tr>`
    ).join("");

    card.innerHTML = `
      <h3>Group ${g}</h3>
      <table>
        <thead><tr><th>Team</th><th class="num">xP</th><th class="num">xGF</th><th class="num">xGA</th><th class="num">xGD</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <details class="matches-detail">
        <summary>Expected match probabilities</summary>
        <table><tbody>${matchRows}</tbody></table>
      </details>
    `;
    container.appendChild(card);
  }
  document.getElementById("groups-count").textContent =
    `${Object.keys(groups).length} groups · 4 teams each · top 2 + 8 best 3rd advance`;
}

function renderThirds(model) {
  const qualified = DATA[model].qualified;  // list of [team, group, seed]
  const container = document.getElementById("thirds");
  container.innerHTML = "";
  const thirds = qualified.filter(q => q[2] === 3);
  for (const [team, g] of thirds) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `<span>${team}</span><span class="g">Grp ${g}</span>`;
    container.appendChild(chip);
  }
}

function renderBracket(model) {
  const rounds = DATA[model].rounds;
  const container = document.getElementById("bracket");
  container.innerHTML = "";
  for (const r of rounds) {
    const col = document.createElement("div");
    col.className = "round-col";
    const header = document.createElement("h3");
    header.textContent = r.round;
    col.appendChild(header);
    for (const m of r.matches) {
      const match = document.createElement("div");
      match.className = "match";
      const aWin = m.winner === m.team_a;
      const bWin = m.winner === m.team_b;
      const mno = m.match_no ? `<div class="match-label">M${m.match_no}</div>` : '';
      match.innerHTML = `
        ${mno}
        <div class="match-row ${aWin ? 'winner' : ''}">
          <span>${m.team_a}</span>
          <span class="prob">${aWin ? pct(m.p_winner) : ''}</span>
        </div>
        <div class="match-row ${bWin ? 'winner' : ''}">
          <span>${m.team_b}</span>
          <span class="prob">${bWin ? pct(m.p_winner) : ''}</span>
        </div>
      `;
      col.appendChild(match);
    }
    container.appendChild(col);
  }
}

function render(model) {
  if (!DATA[model]) {
    document.getElementById("champ").textContent = "(not available)";
    document.getElementById("meta").textContent = `${model.toUpperCase()} JSON missing — rerun predict.py --model ${model}`;
    document.getElementById("groups").innerHTML = "";
    document.getElementById("thirds").innerHTML = "";
    document.getElementById("bracket").innerHTML = "";
    return;
  }
  document.getElementById("champ").textContent = DATA[model].champion;
  document.getElementById("meta").textContent = `${model.toUpperCase()} match probabilities · QAOA knockout optimisation`;
  renderGroups(model);
  renderThirds(model);
  renderBracket(model);
}

document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    render(t.dataset.model);
  });
});

render("xgb");
</script>
</body>
</html>
"""


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for key, path in [("xgb", XGB_JSON), ("qsvm", QSVM_JSON)]:
        if path.exists():
            with path.open() as f:
                pred = json.load(f)
            if "groups" in pred:
                data[key] = pred
            else:
                print(f"WARN: {path.name} missing 'groups' field — rerun predict.py")
                data[key] = None
        else:
            print(f"WARN: {path.name} not found — that model tab will show a placeholder")
            data[key] = None

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    OUTPUT_HTML.write_text(html)
    print(f"Wrote {OUTPUT_HTML}")
    print(f"Open in browser: file://{OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    build()
