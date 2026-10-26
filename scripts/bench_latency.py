"""Latency benchmark (CPU). Model load, warm-up and measurement are kept separate.

 - batch-size-1 request latency: one text per forward call, p50/p95 over N prompts (original condition).
 - batch throughput: texts/second at batch size B; NOT converted to per-request latency.
 - normalization overhead: wall time of normalize() per input, measured on its own.
Uses the DEV split only. Results: results/latency/latency.json
"""
import argparse
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval.data import load_manifest  # noqa: E402
from opir_guardrail_eval.models import GLiClassAdapter  # noqa: E402
from opir_guardrail_eval.normalize import normalize  # noqa: E402
from opir_guardrail_eval.transforms import TRANSFORMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["opir-edge", "opir-large"])
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--device", default="cpu")
ap.add_argument("--tag", default="cpu", help="output file suffix, keeps CPU and GPU results separate")
ap.add_argument("--batch-size", type=int, default=8)
a = ap.parse_args()

import torch  # noqa: E402


def sync():
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()


samples = load_manifest(ROOT / "data" / "manifests" / "dev.jsonl")[: a.n]
texts = [s["text"] for s in samples]
out = {"hardware": {"processor": platform.processor(), "platform": platform.platform(), "device": a.device,
                    "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
                    "torch": torch.__version__, "threads": torch.get_num_threads(), "dtype": "float32"},
       "n_prompts": len(texts), "models": {}}
norm_in = {c: [TRANSFORMS[c](t)[0] for t in texts] for c in TRANSFORMS}
out["normalization_seconds_per_input"] = {}
for c, xs in norm_in.items():
    ts = []
    for x in xs:
        t0 = time.perf_counter()
        normalize(x)
        ts.append(time.perf_counter() - t0)
    out["normalization_seconds_per_input"][c] = {"median": st.median(ts), "p95": sorted(ts)[int(0.95 * len(ts)) - 1], "max": max(ts)}

for name in a.models:
    m = GLiClassAdapter(name, device=a.device, batch_size=1).load()
    lens = [m.token_stats(t)["full_input_tokens"] for t in texts]
    for t in texts[:3]:  # warm-up excluded
        m.predict([t])
    sync()
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        m.predict([t])
        sync()
        lat.append(time.perf_counter() - t0)
    m.batch_size = a.batch_size
    m.predict(texts[: a.batch_size])  # warm-up
    sync()
    t0 = time.perf_counter()
    m.predict(texts)
    sync()
    wall = time.perf_counter() - t0
    out["models"][name] = {
        "load_seconds_cached_weights": m.load_seconds, "mean_full_input_tokens": sum(lens) / len(lens),
        "batch1_latency_seconds": {"p50": st.median(lat), "p95": sorted(lat)[int(0.95 * len(lat)) - 1], "mean": st.mean(lat)},
        "batch_throughput": {"batch_size": a.batch_size, "texts": len(texts), "wall_seconds": wall,
                             "texts_per_second": len(texts) / wall},
        "note": f"{a.device}, fp32; original (unobfuscated) short prompts only"}
    print(name, out["models"][name])
(ROOT / "results" / "latency").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "latency" / f"latency_{a.tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
