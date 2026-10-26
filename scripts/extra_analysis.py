"""Extra diagnostics from saved final predictions: normalization round-trip fidelity, unknown-token
ratios, FP/FN overlap between models and representative errors. Writes results/gpu/final/extra.json.

  python scripts/extra_analysis.py --run gpu --stage final
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval.data import load_manifest  # noqa: E402
from opir_guardrail_eval.metrics import load_predictions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gpu")
ap.add_argument("--stage", default="final")
a = ap.parse_args()
rd = ROOT / "results" / a.run / a.stage
man = {s["sample_id"]: s for s in load_manifest(ROOT / "data" / "manifests" / f"{a.stage}.jsonl")}
out = {}
preds = {}
for m in ("opir-edge", "opir-large"):
    rows = load_predictions(rd / f"predictions_{m}.jsonl")
    preds[m] = rows
    res = {}
    # 1) round-trip fidelity: does normalization of a transformed prompt reproduce the original text?
    exact, bad = Counter(), []
    tot = Counter()
    for r in rows:
        if r["normalized"] and r["condition"] != "original":
            tot[r["condition"]] += 1
            if r["input_sha"] == r["original_sha"]:
                exact[r["condition"]] += 1
            else:
                bad.append({"sample_id": r["sample_id"], "condition": r["condition"], "notes": r["norm_notes"]})
    res["normalization_exact_roundtrip"] = {c: f"{exact[c]}/{tot[c]}" for c in tot}
    res["normalization_inexact_cases"] = bad
    # 2) unknown tokens / token counts, no-normalization rows
    unk = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if not r["normalized"] and "error" not in r:
            u = unk[r["condition"]]
            u[0] += r["n_tokens"]
            u[1] += r["n_unk"] or 0
            u[2] += (r["n_unk"] or 0) > 0
    res["token_stats_no_norm"] = {c: {"tokens": v[0], "unk_tokens": v[1], "unk_ratio": v[1] / v[0], "inputs_with_unk": v[2]}
                                  for c, v in unk.items()}
    res["max_full_input_tokens"] = max(r["full_input_tokens"] for r in rows if "error" not in r)
    out[m] = res

# 3) original-condition errors and cross-model overlap
orig = {m: {r["sample_id"]: r for r in preds[m] if r["condition"] == "original" and not r["normalized"]} for m in preds}
fp = {m: {k for k, r in orig[m].items() if r["label"] == "benign" and r["pred_unsafe"]} for m in orig}
fn = {m: {k for k, r in orig[m].items() if r["label"] == "unsafe" and not r["pred_unsafe"]} for m in orig}
out["overlap"] = {"fp_edge": len(fp["opir-edge"]), "fp_large": len(fp["opir-large"]), "fp_both": len(fp["opir-edge"] & fp["opir-large"]),
                  "fn_edge": len(fn["opir-edge"]), "fn_large": len(fn["opir-large"]), "fn_both": len(fn["opir-edge"] & fn["opir-large"])}
cats = Counter(man[k]["meta"]["category"] for k in fp["opir-large"])
tot_cat = Counter(s["meta"]["category"] for s in man.values() if s["label"] == "benign")
out["large_fp_by_jbb_category"] = {c: f"{cats[c]}/{tot_cat[c]}" for c in tot_cat}
cats = Counter(man[k]["meta"]["category"] for k in fp["opir-edge"])
out["edge_fp_by_jbb_category"] = {c: f"{cats[c]}/{tot_cat[c]}" for c in tot_cat}
out["examples"] = {
    "missed_unsafe_original": {m: [(k, man[k]["text"]) for k in sorted(fn[m])] for m in fn},
    "benign_blocked_original_large_first5": [(k, man[k]["text"]) for k in sorted(fp["opir-large"])[:5]],
    "benign_clean_large_first5": [(k, man[k]["text"]) for k in sorted(set(orig["opir-large"]) - fp["opir-large"] - {x for x in orig["opir-large"] if man[x]["label"] == "unsafe"})[:5]],
}
# 4) top score distribution: do flagged benign prompts have high scores? (descriptive only, scores not calibrated)
for m in preds:
    sc = [r["top_score"] for k, r in orig[m].items() if k in fp[m]]
    out[m]["mean_top_score_on_benign_false_positives"] = sum(sc) / len(sc) if sc else None
(rd / "extra.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
