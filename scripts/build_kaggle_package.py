"""Build the private Kaggle upload package under kaggle/ (dataset = project code + public manifests,
kernel = notebook). Excludes credentials, caches, venv, model weights, raw downloads and prior results.

  python scripts/build_kaggle_package.py --username <kaggle-username>
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--username", required=True)
ap.add_argument("--dataset-slug", default="opir-guardrail-eval-project")
ap.add_argument("--kernel-slug", default="opir-guardrail-eval-gpu")
a = ap.parse_args()

pkg = ROOT / "kaggle"
if pkg.exists():
    shutil.rmtree(pkg)
ds, kn = pkg / "dataset", pkg / "kernel"
ds.mkdir(parents=True)
kn.mkdir(parents=True)
INCLUDE = ["src", "scripts", "tests", "configs", "docs", "pyproject.toml", "data/manifests"]
for item in INCLUDE:
    src = ROOT / item
    dst = ds / item
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
shutil.rmtree(ds / "scripts" / "__pycache__", ignore_errors=True)
(ds / "dataset-metadata.json").write_text(json.dumps({
    "title": "opir guardrail eval project", "id": f"{a.username}/{a.dataset_slug}",
    "licenses": [{"name": "other"}], "isPrivate": True,
    "description": "Private: evaluation code and public-data manifests (JBB-Behaviors MIT, XSTest CC-BY-4.0)."}, indent=2))

S = lambda src: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}  # noqa: E731
M = lambda src: {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}  # noqa: E731
cells = [
    M("# Opir guardrail robustness evaluation (GPU, free Kaggle quota)\nRuns tests, pilot, dev and final stages **one model at a time** with the frozen settings. Resumable: re-running a cell reuses `results/cache`."),
    S(f"""
import os, subprocess, sys, shutil, json, time, glob
DATASET = "{a.dataset_slug}"
SRC = glob.glob(f"/kaggle/input/**/{{DATASET}}", recursive=True) or glob.glob("/kaggle/input/*")
SRC = [p for p in SRC if os.path.exists(os.path.join(p, "pyproject.toml"))][0]
WORK = "/kaggle/working/opir-guardrail-eval"
if not os.path.exists(WORK):
    shutil.copytree(SRC, WORK)
os.chdir(WORK)
os.makedirs("logs", exist_ok=True)
os.environ["HF_HOME"] = "/tmp/hf"          # keep model weights out of /kaggle/working outputs
os.environ["PYTHONUNBUFFERED"] = "1"
BATCH = int(os.environ.get("OPIR_BATCH", "16"))
def sh(cmd, log):
    print(">>", cmd, flush=True)
    with open(f"logs/{{log}}.log", "a") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line); f.flush()
            if not any(w in line for w in ("Warning", "Loading weights")): print(line, end="")
        p.wait()
    print("exit", p.returncode, flush=True)
    return p.returncode
"""),
    S("""
sh("nvidia-smi", "env")
sh("pip install -q gliclass 'transformers>=5.0' pytest", "env")
sh("pip freeze > logs/pip_freeze.txt", "env")
import torch
assert torch.cuda.is_available(), "No GPU: enable a GPU accelerator in notebook settings"
cap = torch.cuda.get_device_capability(0)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), cap)
x = torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda"); torch.cuda.synchronize()  # kernel-compat check
"""),
    S("""
rc = sh("python -m pytest -q tests", "tests"); assert rc == 0, "tests failed"
"""),
    S("""
# prepare_data re-fetches the pinned public datasets and verifies they reproduce the shipped manifests
before = {p: open(p, "rb").read() for p in ("data/manifests/final.jsonl", "data/manifests/dev.jsonl")}
assert sh("python scripts/prepare_data.py", "data") == 0
assert all(open(p, "rb").read() == b for p, b in before.items()), "manifests differ from shipped ones"
print("manifests reproduced identically")
"""),
    M("## Pilot (dev split, 10+10) — one model at a time"),
    S("""
for m in ["opir-edge", "opir-large"]:
    assert sh(f"python scripts/run_eval.py --stage pilot --models {m} --device cuda:0 --batch-size {BATCH}", f"pilot_{m}") == 0
"""),
    M("## Dev (30+30) then final (100+100) — one model at a time"),
    S("""
for stage in ["dev", "final"]:
    for m in ["opir-edge", "opir-large"]:
        assert sh(f"python scripts/run_eval.py --stage {stage} --models {m} --device cuda:0 --batch-size {BATCH}", f"{stage}_{m}") == 0
"""),
    S("""
sh(f"python scripts/bench_latency.py --device cuda:0 --tag gpu --batch-size {BATCH}", "latency_gpu")
for stage in ["pilot", "dev", "final"]:
    sh(f"python scripts/make_metrics.py --stage {stage}", f"metrics_{stage}")
shutil.make_archive("/kaggle/working/opir_results", "zip", WORK, "results")
shutil.copytree("logs", "/kaggle/working/logs", dirs_exist_ok=True)
print(os.listdir("/kaggle/working"))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(kn / "opir_guardrail_eval_gpu.ipynb").write_text(json.dumps(nb, indent=1))
(kn / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.username}/{a.kernel_slug}", "title": "opir guardrail eval gpu",
    "code_file": "opir_guardrail_eval_gpu.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [f"{a.username}/{a.dataset_slug}"], "competition_sources": [], "kernel_sources": [],
    "model_sources": []}, indent=2))
print("package at", pkg)
