"""Evaluation runner: builds paired (sample x transform x normalization) inputs, runs a model with a
resumable on-disk prediction cache, and writes one predictions row per evaluated condition."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

from .models import GLiClassAdapter, text_hash
from .normalize import normalize
from .transforms import CONDITIONS_ORDER, TRANSFORMS


def build_rows(samples: list[dict]) -> list[dict]:
    """One row per (sample, transform, normalize flag). Normalization time measured per input."""
    rows = []
    for s in samples:
        for cond in CONDITIONS_ORDER:
            tx, params = TRANSFORMS[cond](s["text"])
            t0 = time.perf_counter()
            nr = normalize(tx)
            norm_s = time.perf_counter() - t0
            for use_norm in (False, True):
                inp = nr.normalized if use_norm else tx
                rows.append({
                    "sample_id": s["sample_id"], "split": s["split"], "label": s["label"], "source": s["source"],
                    "condition": cond, "normalized": use_norm, "transform_params": params,
                    "original_len": len(s["text"]), "transformed_len": len(tx), "input_len": len(inp),
                    "original_sha": text_hash(s["text"]), "transformed_sha": text_hash(tx), "input_sha": text_hash(inp),
                    "norm_fired": nr.fired if use_norm else [], "norm_notes": nr.notes if use_norm else [],
                    "norm_changed": nr.changed if use_norm else False, "norm_seconds": norm_s if use_norm else 0.0,
                    "_input": inp,
                })
    return rows


class PredCache:
    """Append-only jsonl keyed by input hash (the file name encodes model revision + settings)."""

    def __init__(self, path: Path):
        self.path, self.data = path, {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated last line from an interrupted run
                    self.data[r["key"]] = r  # later lines win

    def todo(self, keys):
        return [k for k in keys if k not in self.data or "error" in self.data[k]]  # errors are retried

    def add(self, rec):
        self.data[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(samples, adapter: GLiClassAdapter, out_dir: Path, cache_root: Path, stage: str, log=print) -> dict:
    rows = build_rows(samples)
    uniq = {}
    for r in rows:
        uniq.setdefault(r["input_sha"], r["_input"])
    cache = PredCache(cache_root / f"{adapter.name}_{adapter.revision[:8]}_{adapter.settings_key()}.jsonl")
    todo = cache.todo(list(uniq))
    log(f"[{adapter.name}] rows={len(rows)} unique_inputs={len(uniq)} to_infer={len(todo)} (cached={len(uniq) - len(todo)})")
    if adapter.pipe is None:
        adapter.load()  # tokenizer stats are needed for all rows
    # token stats for cached entries lacking them are already stored; only new ones are computed
    todo.sort(key=lambda k: len(uniq[k]))
    t_start, done, bs = time.perf_counter(), 0, adapter.batch_size
    for i in range(0, len(todo), bs):
        keys = todo[i:i + bs]
        texts = [uniq[k] for k in keys]
        try:
            for k, t, p in zip(keys, texts, adapter.predict(texts)):
                cache.add({"key": k, **p, **adapter.token_stats(t)})
        except Exception:  # never treat a failure as "safe"; retry singly to isolate the bad input
            for k, t in zip(keys, texts):
                try:
                    cache.add({"key": k, **adapter.predict([t])[0], **adapter.token_stats(t)})
                except Exception as e2:
                    cache.add({"key": k, "error": f"{type(e2).__name__}: {e2}"[:500]})
        done += len(keys)
        if (i // bs) % 10 == 0:
            log(f"  {done}/{len(todo)} elapsed {time.perf_counter() - t_start:.0f}s")
    infer_wall = time.perf_counter() - t_start if todo else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"predictions_{adapter.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r.pop("_input")
            c = cache.data.get(r["input_sha"], {"error": "missing"})
            r.update({"model": adapter.name, "revision": adapter.revision, "stage": stage})
            if "error" in c:
                r["error"] = c["error"]
            else:
                r.update({k: c[k] for k in ("top_label", "top_score", "pred_unsafe", "raw", "n_tokens", "n_unk",
                                            "full_input_tokens", "truncated")})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"stage": stage, "model": adapter.name, "settings": adapter.settings(), "n_samples": len(samples),
            "n_rows": len(rows), "n_unique_inputs": len(uniq), "n_newly_inferred": len(todo),
            "model_load_seconds": adapter.load_seconds, "inference_wall_seconds_this_run": infer_wall,
            "batch_size": adapter.batch_size, "python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        import transformers
        meta.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
                    cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    gpu_capability=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                    gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None)
    except Exception:
        pass
    (out_dir / f"run_meta_{adapter.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
