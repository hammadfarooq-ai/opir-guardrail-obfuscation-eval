"""Download public datasets (pinned revisions) and write frozen manifests. Usage: python scripts/prepare_data.py"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opir_guardrail_eval.data import SOURCES, write_manifests  # noqa: E402


def fetch(repo, rev, path, dest):
    url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


raw = ROOT / "data" / "raw"
j, x = SOURCES["jbb"], SOURCES["xstest"]
fetch(j["repo"], j["revision"], j["files"]["harmful"], raw / "jbb_harmful.csv")
fetch(j["repo"], j["revision"], j["files"]["benign"], raw / "jbb_benign.csv")
fetch(x["repo"], x["revision"], x["files"]["all"], raw / "xstest_prompts.csv")
print(write_manifests(raw, ROOT / "data" / "manifests"))
