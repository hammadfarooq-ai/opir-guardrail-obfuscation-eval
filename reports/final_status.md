# Final status (2026-10-26)

## Completed
- Verified sources (docs/sources.md): both Opir checkpoints public, ungated, Apache-2.0, no custom remote code; JBB-Behaviors (MIT) and XSTest (CC-BY-4.0, via ungated `Paul/XSTest`).
- Pipeline: transforms, auditable normalizer, model adapter with resumable cache and error recording, metrics (Wilson, paired bootstrap), figures, reports, README, docs.
- **Full evaluation executed on Kaggle (free GPU, private notebook)**: both models, one at a time, pilot (10+10) -> dev (30+30) -> final (100+100), all 5 conditions x {no norm, norm}. 0 inference errors, 0 truncations. 10 conditions per model (5 input forms x 2 normalization settings; an earlier mention of 12 was an error), 2000 rows and 1002 unique inputs per model (reconciliation in evaluation_report.md section 2.1).
- Kaggle: kernel `malikhammadfarooq/opir-guardrail-eval-gpu` (private), version 1, status COMPLETE (verified with `kaggle kernels status`), URL https://www.kaggle.com/code/malikhammadfarooq/opir-guardrail-eval-gpu . GPU: Tesla T4 (2 available; cuda:0 used), CUDA 12.8 build, torch 2.11.0+cu128, transformers 5.16.1, gliclass 0.1.21, fp32, batch size 16. End-to-end final-stage loop time on the T4 (includes token statistics and cache writes, not model-only throughput; see evaluation_report.md section 6.1): 1.9 s (edge) and 22.2 s (large) for 1002 unique inputs; total notebook runtime was under about 5 minutes (pushed ~15:05 and seen COMPLETE at 15:10 by local-clock status polling; nvidia-smi in the run logged 10:05 on the Kaggle clock; exact runtime not recorded by the kernel API).
- Local: CPU pilot (10+10) for both models with the frozen code, CPU latency benchmark; CPU and GPU pilots agree (0 label disagreements, max score difference < 5e-5).
- Outputs downloaded to `results/gpu/` (predictions, run metadata, logs, cache, pip freeze, latency) and `results/kaggle_run1/` (kernel log + results zip).

## Checks performed before launch
- The earlier broken `data.py` commit (499f0f2) was fixed in `0e2f19e`; tag `frozen-settings-v1` was re-pointed (local tag only) to `0e2f19e` and verified; the tagged `data.py` parses; tests pass (13 locally).
- Dataset hashes: Kaggle's dataset download endpoint returned 404 for the private dataset, so verification was done from inside the Kaggle run: the notebook's working copy (17 files) is sha256-identical to the local package, and the notebook re-fetched the public data and reproduced the shipped manifests byte for byte (assertion passed).
- GPU: nvidia-smi showed 2x Tesla T4; the notebook asserts `torch.cuda.is_available()` and runs a CUDA matmul.
- Secret scan of repository, notebook, logs: only revision hashes matched; no credential files tracked. Kaggle credentials are read from the CLI's own config and were never printed.

## Commands actually executed (summary)
Environment setup (`python -m venv`, `pip install torch gliclass transformers pandas matplotlib numpy pytest`), `scripts/prepare_data.py`, `pytest -q tests` (several times), `scripts/run_eval.py --stage smoke|pilot` (local CPU), `scripts/bench_latency.py --device cpu`, `scripts/build_kaggle_package.py`, `kaggle datasets create/version`, `kaggle kernels push/status/output`, `scripts/make_metrics.py --stage pilot|dev|final [--run gpu]`, `scripts/extra_analysis.py`.

## Tests
`python -m pytest -q tests`: **13 passed, 0 skipped** locally (incl. a new always-running manifest isolation test; transforms, round trips, bounded/strict base64, no-change on clean text, split isolation/label mapping/determinism, metric computation incl. errors and paired recovery, bootstrap, mocked adapter and resume). On Kaggle: 11 passed, 1 skipped (the data-dependent split test skips when raw data is absent; raw data was fetched after the test cell). The new manifest test was not run on Kaggle; Kaggle's check was the byte-identical manifest reproduction.

## Not done / blocked / caveats
- Local CPU **final** and **dev** runs were not performed (GPU run covers them). The CPU latency benchmark was completed (results/latency/latency_cpu.json) and is reported separately from GPU timings.
- Audit (after the run): headline 54/100 and 71/100 FPR verified from raw predictions; no bug found, no code changed after the freeze; see evaluation_report.md section 2.1. The FPR is the observed flag count under the dataset's benign labels; possible policy mismatch is an untested interpretation. Pilot, dev and final ran automatically in one Kaggle sequence with no manual inspection gate; no setting was changed using final results.
- `pip install -e ".[dev]"` and `pip install -r requirements.lock.txt` were not re-tested from a clean environment (the lock file has a `+cpu` torch pin that needs the PyTorch CPU index).
- Two normalization round-trips were inexact (documented, not patched after freezing).
- `results/pilot_cpu_prefreeze_normalizer/` is an earlier CPU pilot output generated before the spaced-rule fix; retained for transparency, not used in reports.
- Kaggle's CPU/GPU numbers are not an official benchmark; no timing comparison with CUDA at other batch sizes.
- An MIT LICENSE file for the repository's own code was added at release preparation (dataset licenses are in THIRD_PARTY_NOTICES.md).
- Nothing was published, pushed, or sent; the message to Ihor is a draft only.

## Deliverable paths (relative to the repository root)
- `README.md`, `THIRD_PARTY_NOTICES.md`, `reports/evaluation_report.md`, `reports/final_status.md`; the message draft `reports/ihor_message_draft.md` is kept locally and untracked (not part of the release)
- `docs/sources.md`, `docs/methodology.md`, `docs/limitations.md`
- `results/gpu/final/` (predictions_*.jsonl, run_meta_*.json, metrics.json, tables.md, extra.json), `results/gpu/dev`, `results/gpu/pilot`, `results/gpu/latency_gpu.json`, `results/latency/latency_cpu.json`, `results/pilot/` (local CPU)
- `reports/figures/gpu_final_{recall,fpr,normalization_effect}.png`
- `data/manifests/`, `requirements.lock.txt`, `results/environment_local.json`, `results/gpu/pip_freeze.txt`
- Git: local repository, tag `frozen-settings-v1` -> `0e2f19e96468c071e291d9b2acba9d4f51addfa7` (inference code, run_eval.py and manifests unchanged since); later commits add results, reports, diagnostics and one test.
- Per-prompt predictions for inspection: `results/gpu/final/original_text_predictions.csv`

## Git history note
Before publication the local commit author/committer identity was rewritten to the project owner's GitHub noreply address (`git filter-branch`; all 11 commits and the `frozen-settings-v1` tag were rewritten). Every rewritten commit has a byte-identical tree to its original (verified commit by commit), so code, data and results are unchanged; only commit hashes changed, and the hashes quoted in these documents are the rewritten ones. The pre-rewrite history is kept only in a backup bundle outside the repository.

## Remaining manual steps
1. Review `reports/` and decide whether to publish the repository and replace `[repository link]` in the locally kept `reports/ihor_message_draft.md`.
2. Optionally delete the private Kaggle kernel/dataset when no longer needed.
3. Send the message yourself.
