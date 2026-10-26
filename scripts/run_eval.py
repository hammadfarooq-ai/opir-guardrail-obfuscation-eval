"""Run an evaluation stage (re-running the same command resumes from the prediction cache).

  python scripts/run_eval.py --stage smoke --models opir-edge
  python scripts/run_eval.py --stage pilot --models opir-edge opir-large
  python scripts/run_eval.py --stage dev   --models opir-edge opir-large
  python scripts/run_eval.py --stage final --models opir-edge opir-large

smoke/pilot draw from the DEV split so the final split stays untouched until settings are frozen.
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval.data import SEED, load_manifest  # noqa: E402
from opir_guardrail_eval.models import GLiClassAdapter  # noqa: E402
from opir_guardrail_eval.runner import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["smoke", "pilot", "dev", "final"])
ap.add_argument("--models", nargs="+", default=["opir-edge", "opir-large"])
ap.add_argument("--device", default="cpu")
ap.add_argument("--batch-size", type=int, default=4)
a = ap.parse_args()

m = ROOT / "data" / "manifests"
if a.stage == "final":
    samples = load_manifest(m / "final.jsonl")
else:
    samples = load_manifest(m / "dev.jsonl")
    if a.stage in ("smoke", "pilot"):
        n = 2 if a.stage == "smoke" else 10
        rng = random.Random(SEED)
        pick = []
        for lab in ("unsafe", "benign"):
            pick += rng.sample([s for s in samples if s["label"] == lab], n)
        samples = pick
for name in a.models:
    run(samples, GLiClassAdapter(name, device=a.device, batch_size=a.batch_size),
        ROOT / "results" / a.stage, ROOT / "results" / "cache", a.stage)
