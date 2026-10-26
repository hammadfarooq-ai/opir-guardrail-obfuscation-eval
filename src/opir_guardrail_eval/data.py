"""Dataset preparation: public labelled prompts -> frozen manifests with fixed seed.

Final evaluation: JailbreakBench JBB-Behaviors (MIT): 100 harmful (unsafe) + 100 benign.
Development:      XSTest (CC-BY-4.0, Paul/XSTest): 30 safe + 30 unsafe, deduplicated against final.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

SEED = 20261106
SOURCES = {
    "jbb": {"repo": "JailbreakBench/JBB-Behaviors", "revision": "886acc352a31533ffbcf4ef22c744658688086fc",
            "license": "MIT", "files": {"harmful": "data/harmful-behaviors.csv",
                                         "benign": "data/benign-behaviors.csv"}},
    "xstest": {"repo": "Paul/XSTest", "revision": "f600c994b256f12867dfa5b3eb3d545a3e62f8b5",
               "license": "CC-BY-4.0", "files": {"all": "xstest_prompts.csv"}},
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def canon(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower().rstrip(".?!")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jbb(raw: Path) -> list[dict]:
    rows = []
    for label, fname in (("unsafe", "jbb_harmful.csv"), ("benign", "jbb_benign.csv")):
        for r in read_csv(raw / fname):
            rows.append({"sample_id": f"jbb-{'h' if label == 'unsafe' else 'b'}-{int(r['Index']):03d}",
                         "source": "jbb", "label": label, "text": r["Goal"].strip(),
                         "meta": {"category": r["Category"], "behavior": r["Behavior"], "origin": r["Source"]}})
    return rows


def load_xstest(raw: Path) -> list[dict]:
    rows = []
    for r in read_csv(raw / "xstest_prompts.csv"):
        lab = {"safe": "benign", "unsafe": "unsafe"}[r["label"].strip().lower()]
        rows.append({"sample_id": f"xstest-{int(r['id']):03d}", "source": "xstest", "label": lab,
                     "text": r["prompt"].strip(), "meta": {"type": r.get("type", ""), "focus": r.get("focus", "")}})
    return rows


def dedupe(rows: list[dict], seen: set[str] | None = None) -> tuple[list[dict], list[str]]:
    seen = set() if seen is None else seen
    kept, dropped = [], []
    for r in rows:
        k = canon(r["text"])
        if k in seen:
            dropped.append(r["sample_id"])
            continue
        seen.add(k)
        kept.append(r)
    return kept, dropped


def build_splits(raw: Path, n_dev_each: int = 30, seed: int = SEED) -> dict:
    final_rows, dropped_final = dedupe(load_jbb(raw))
    final_keys = {canon(r["text"]) for r in final_rows}
    xs, dropped_dev = dedupe(load_xstest(raw), set(final_keys))
    rng = random.Random(seed)
    dev = []
    for lab in ("unsafe", "benign"):
        pool = sorted([r for r in xs if r["label"] == lab], key=lambda r: r["sample_id"])
        dev += rng.sample(pool, n_dev_each)
    for split, rows in (("final", final_rows), ("dev", dev)):
        for r in rows:
            r["split"] = split
            r["text_sha"] = sha(r["text"])
    assert not ({r["sample_id"] for r in final_rows} & {r["sample_id"] for r in dev})
    assert not ({canon(r["text"]) for r in final_rows} & {canon(r["text"]) for r in dev})
    return {"final": final_rows, "dev": dev,
            "dropped": {"final_duplicates": dropped_final, "dev_duplicates_or_overlap": dropped_dev}}


def write_manifests(raw: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    s = build_splits(raw)
    for split in ("final", "dev"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in s[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"seed": SEED, "sources": SOURCES, "dropped": s["dropped"],
               "counts": {sp: {lab: sum(r["label"] == lab for r in s[sp]) for lab in ("unsafe", "benign")}
                          for sp in ("final", "dev")},
               "label_mapping": {"jbb harmful split": "unsafe", "jbb benign split": "benign",
                                 "xstest label=unsafe": "unsafe", "xstest label=safe": "benign"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
