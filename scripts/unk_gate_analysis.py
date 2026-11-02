"""Descriptive check on saved predictions: how often does the tokenizer emit [UNK], and would an
"any [UNK] in the input" gate have flagged the missed unsafe prompts? No model is run.

  python scripts/unk_gate_analysis.py --run gpu --stage final
Writes results/<run>/<stage>/unk_gate.json and prints a table.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval.metrics import load_predictions  # noqa: E402
from opir_guardrail_eval.transforms import CONDITIONS_ORDER  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gpu")
ap.add_argument("--stage", default="final")
a = ap.parse_args()
rd = ROOT / "results" / a.run / a.stage
out = {}
for m in ("opir-large", "opir-edge"):
    rows = load_predictions(rd / f"predictions_{m}.jsonl")
    res = {"no_normalization": {}}
    for cond in CONDITIONS_ORDER:
        r = [x for x in rows if x["condition"] == cond and not x["normalized"] and "error" not in x]
        with_unk = [x for x in r if (x["n_unk"] or 0) > 0]
        missed = [x for x in r if x["label"] == "unsafe" and not x["pred_unsafe"]]
        shares = [x["n_unk"] / x["n_tokens"] for x in with_unk]
        res["no_normalization"][cond] = {
            "inputs": len(r), "inputs_with_unk": len(with_unk),
            "unsafe_with_unk": sum(x["label"] == "unsafe" for x in with_unk),
            "benign_with_unk": sum(x["label"] == "benign" for x in with_unk),
            "median_unk_share_of_tokens_among_those": st.median(shares) if shares else None,
            "missed_unsafe": len(missed), "missed_unsafe_with_unk": sum((x["n_unk"] or 0) > 0 for x in missed)}
    rn = [x for x in rows if x["normalized"] and "error" not in x]
    res["with_normalization_inputs_with_unk"] = sum((x["n_unk"] or 0) > 0 for x in rn)
    res["with_normalization_inputs"] = len(rn)
    out[m] = res
(rd / "unk_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for m, res in out.items():
    print("==", m)
    for cond, v in res["no_normalization"].items():
        print(f"  {cond:10s} [UNK] in {v['inputs_with_unk']}/{v['inputs']} inputs "
              f"(unsafe {v['unsafe_with_unk']}, benign {v['benign_with_unk']}); missed unsafe {v['missed_unsafe']}, with [UNK] {v['missed_unsafe_with_unk']}")
    print(f"  with normalization: [UNK] in {res['with_normalization_inputs_with_unk']}/{res['with_normalization_inputs']} inputs")
