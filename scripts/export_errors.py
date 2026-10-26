"""Export per-prompt original-text predictions for the final split (JBB, MIT) to CSV for manual inspection.
  python scripts/export_errors.py --run gpu --stage final
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval.data import load_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gpu")
ap.add_argument("--stage", default="final")
a = ap.parse_args()
rd = ROOT / "results" / a.run / a.stage
man = {s["sample_id"]: s for s in load_manifest(ROOT / "data" / "manifests" / f"{a.stage}.jsonl")}
pred = {}
for m in ("opir-large", "opir-edge"):
    for line in open(rd / f"predictions_{m}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] == "original" and not r["normalized"]:
            pred[(m, r["sample_id"])] = r
with open(rd / "original_text_predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "label", "category", "text", "large_pred", "large_top_score", "edge_pred", "edge_top_score"])
    for sid, s in man.items():
        L, E = pred[("opir-large", sid)], pred[("opir-edge", sid)]
        w.writerow([sid, s["label"], s["meta"]["category"], s["text"], L["top_label"], round(L["top_score"], 4),
                    E["top_label"], round(E["top_score"], 4)])
print("wrote", rd / "original_text_predictions.csv")
