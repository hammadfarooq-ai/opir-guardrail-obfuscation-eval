"""Regenerate metrics JSON, markdown tables and figures purely from saved predictions.

  python scripts/make_metrics.py --stage final
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval import metrics  # noqa: E402
from opir_guardrail_eval.transforms import CONDITIONS_ORDER  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
a = ap.parse_args()
rd = ROOT / "results" / a.stage
fig_dir = ROOT / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
models = sorted(p.stem.replace("predictions_", "") for p in rd.glob("predictions_*.jsonl"))
allm, md = {}, []
pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
ci = lambda c: "" if c[0] is None else f" [{100 * c[0]:.0f}-{100 * c[1]:.0f}]"  # noqa: E731

for m in models:
    rows = metrics.load_predictions(rd / f"predictions_{m}.jsonl")
    s = metrics.summarize(rd / f"predictions_{m}.jsonl")
    # normalization effect on clean benign originals (damage inspection)
    damaged = [{"sample_id": r["sample_id"], "fired": r["norm_fired"], "notes": r["norm_notes"]}
               for r in rows if r["condition"] == "original" and r["normalized"] and r["norm_changed"]]
    s["clean_original_text_changed_by_normalization"] = damaged
    s["failed_inferences"] = [{"sample_id": r["sample_id"], "condition": r["condition"], "normalized": r["normalized"],
                               "error": r["error"]} for r in rows if "error" in r]
    allm[m] = s
    md += [f"### {m} ({a.stage})", "",
           "| condition | norm | unsafe n | benign n | TP | FN | TN | FP | recall [95% Wilson] | FPR [95% Wilson] | precision | F1 | errors | trunc | mean tok | unk inputs |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["per_condition"]:
        md.append(f"| {r['condition']} | {'yes' if r['normalized'] else 'no'} | {r['n_unsafe']} | {r['n_benign']} | {r['tp']} | {r['fn']} | {r['tn']} | {r['fp']} | "
                  f"{pct(r['recall'])}{ci(r['recall_ci'])} | {pct(r['fpr'])}{ci(r['fpr_ci'])} | {pct(r['precision'])} | {pct(r['f1'])} | {r['errors']} | "
                  f"{r['truncated']} | {r['mean_tokens']:.0f} | {r['inputs_with_unk']} |")
    md += ["", "Normalization recovery / regression (paired by sample_id; pairs with an error excluded):", "",
           "| condition | unsafe missed->detected | unsafe detected->missed | benign newly blocked | benign unblocked | unsafe pairs | benign pairs | inputs changed by norm |",
           "|---|---|---|---|---|---|---|---|"]
    for r in s["recovery"]:
        md.append(f"| {r['condition']} | {r['unsafe_missed_then_detected']} | {r['unsafe_detected_then_missed']} | {r['benign_newly_blocked']} | "
                  f"{r['benign_unblocked']} | {r['n_unsafe_pairs']} | {r['n_benign_pairs']} | {r['text_changed_by_norm']} |")
    md += ["", "Paired source-level bootstrap, difference (normalized - not normalized), 95% CI:", "",
           "| condition | metric | pairs | diff | 95% CI |", "|---|---|---|---|---|"]
    for b in s["bootstrap"]:
        md.append(f"| {b['condition']} | {b['metric']} | {b['n_pairs']} | {100 * b['diff']:+.1f} pp | [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}] |")
    md += ["", f"Clean original prompts altered by normalization: {len(damaged)}; failed inferences: {len(s['failed_inferences'])}.", ""]

(rd / "metrics.json").write_text(json.dumps(allm, indent=2), encoding="utf-8")
(rd / "tables.md").write_text("\n".join(md), encoding="utf-8")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

colors = {"opir-large": "#1f4e79", "opir-edge": "#d97706"}


def bars(metric, ci_key, title, fname, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, norm in zip(axes, (False, True)):
        w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            rs = {r["condition"]: r for r in allm[m]["per_condition"] if r["normalized"] == norm}
            xs = [j + i * w for j in range(len(CONDITIONS_ORDER))]
            ys = [100 * (rs[c][metric] or 0) for c in CONDITIONS_ORDER]
            err = [[100 * ((rs[c][metric] or 0) - (rs[c][ci_key][0] or 0)) for c in CONDITIONS_ORDER],
                   [100 * ((rs[c][ci_key][1] or 0) - (rs[c][metric] or 0)) for c in CONDITIONS_ORDER]]
            ax.bar(xs, ys, w, yerr=err, capsize=2, label=m, color=colors.get(m))
        ax.set_xticks([j + w * (len(models) - 1) / 2 for j in range(len(CONDITIONS_ORDER))])
        ax.set_xticklabels(CONDITIONS_ORDER, rotation=20)
        ax.set_title("with normalization" if norm else "no normalization")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle(f"{title} ({a.stage} split; 95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{a.stage}_{fname}.png", dpi=150)
    plt.close(fig)


bars("recall", "recall_ci", "Unsafe recall by condition", "recall", "recall (%)")
bars("fpr", "fpr_ci", "Benign false-positive rate by condition", "fpr", "false-positive rate (%)")

fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), squeeze=False)
for ax, m in zip(axes[0], models):
    rec = allm[m]["recovery"]
    xs = range(len(rec))
    series = [("unsafe_missed_then_detected", "unsafe recovered", "#2e7d32"), ("unsafe_detected_then_missed", "unsafe lost", "#c62828"),
              ("benign_newly_blocked", "benign newly blocked", "#6a1b9a"), ("benign_unblocked", "benign unblocked", "#90a4ae")]
    for k, (key, lab, col) in enumerate(series):
        ax.bar([x + k * 0.2 for x in xs], [r[key] for r in rec], 0.2, label=lab, color=col)
    ax.set_xticks([x + 0.3 for x in xs])
    ax.set_xticklabels([r["condition"] for r in rec], rotation=20)
    ax.set_title(m)
    ax.set_ylabel("number of source prompts")
axes[0][0].legend(fontsize=8)
fig.suptitle(f"Normalization recoveries and regressions, paired by source prompt ({a.stage})")
fig.tight_layout()
fig.savefig(fig_dir / f"{a.stage}_normalization_effect.png", dpi=150)
print("wrote", rd / "metrics.json", rd / "tables.md", "and figures")
