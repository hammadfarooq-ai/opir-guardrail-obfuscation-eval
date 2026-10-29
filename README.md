# opir-guardrail-obfuscation-eval

Independent, exploratory robustness check of two public Knowledgator Opir safety classifiers (`knowledgator/opir-multitask-large-v1.0` and `knowledgator/opir-edge-v1.0`) under common input obfuscations (homoglyphs, zero-width characters, base64, spaced-out letters), with and without a documented normalization step. **Not an official Knowledgator benchmark.**

> Content note: the datasets contain requests for harmful content (text only, no instructions or outputs). Prompts are included under their dataset licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## What was run
- **Final set:** JailbreakBench JBB-Behaviors, 100 harmful (unsafe) + 100 benign prompts (MIT). Each prompt in 5 forms (original, homoglyph, zero-width, base64, spaced) x {no normalization, normalization} = 10 conditions per model, 2000 rows per model, 1002 unique model inputs.
- **Dev set:** XSTest, 30 safe + 30 unsafe prompts (CC-BY-4.0); smoke (2+2) and pilot (10+10) subsets come from it. No thresholds were tuned.
- **Models:** binary `["safe", "unsafe"]`, `classification_type="single-label"`, top label = decision, fp32, max length 1024, no remote code. Large revision `69bb27407d66`, edge revision `467a0431f744`.
- **Hardware:** full run on a free Kaggle Tesla T4 (batch size 16); local CPU pilot (Intel i5-7200U) and a separate CPU latency benchmark.
- 0 inference errors and 0 truncated inputs in the final run. Code frozen at git tag `frozen-settings-v1` (commit `0e2f19e`) before the final run; the pilot, dev and final stages ran automatically in one sequence without a manual inspection gate, and no setting was changed using final results.

## Measured results (final split; counts out of 100 per class, so recall and FPR are also percentages)
Recall = unsafe prompts flagged, FPR = benign prompts flagged (both under the dataset's labels).

| condition | large recall / FPR, no norm | large recall / FPR, norm | edge recall / FPR, no norm | edge recall / FPR, norm |
|---|---|---|---|---|
| original | 97 / 54 | 97 / 54 | 97 / 71 | 97 / 71 |
| homoglyph | 68 / 33 | 97 / 54 | 2 / 0 | 97 / 71 |
| zero-width | 1 / 5 | 97 / 54 | 8 / 11 | 97 / 71 |
| base64 | 81 / 84 | 97 / 54 | 91 / 95 | 97 / 71 |
| spaced | 95 / 93 | 97 / 54 | 100 / 100 | 97 / 71 |

- Normalization is a tradeoff, not an overall improvement. For homoglyph and zero-width input it recovered detection but also brought back the benign blocking seen on unobfuscated text (e.g. edge zero-width: 89 unsafe prompts recovered, 61 benign prompts newly blocked). For base64 and spaced input it mainly removed benign false positives. Full paired counts: [report section 4](reports/evaluation_report.md#4-normalization-effect-paired-by-source-prompt).
- High recall together with a high benign false-positive rate (base64, spaced, no normalization) does not establish that the models recovered the content.
- The 54/100 and 71/100 are flags on the JBB benign split under the dataset's labels, not an estimate of FPR on ordinary traffic. On the 30 XSTest safe dev prompts (a different dataset) the flags were 0/30 (large) and 4/30 (edge). Whether some flags reflect a stricter policy than the dataset's is an untested interpretation.
- **Known-transform limitation:** the normalizer was written for exactly the four transforms generated here, and `normalize(transform(x))` reproduced the original prompt for 398/400 transformed prompts (a text-processing property measured without any model). The results therefore show recovery on known synthetic transforms, not robustness to unseen obfuscations or attacks.
- Tokenizer (descriptive): the large model's tokenizer emitted `[UNK]` on the zero-width inputs (200/200); the edge tokenizer emitted none. This does not establish the cause of the prediction changes.

## Reports and documents
- [Evaluation report](reports/evaluation_report.md) (results, audit of the headline numbers, timing audit, limitations, next experiments)
- [Final status](reports/final_status.md) (what was executed, tests, blockers)
- [Sources and verified revisions](docs/sources.md), [Methodology](docs/methodology.md), [Limitations](docs/limitations.md)
- Figures: [recall](reports/figures/gpu_final_recall.png), [false-positive rate](reports/figures/gpu_final_fpr.png), [normalization effect](reports/figures/gpu_final_normalization_effect.png)
- Machine-readable results: `results/gpu/final/` (predictions, run metadata, metrics, tables), per-prompt original-text predictions in `results/gpu/final/original_text_predictions.csv`

![recall by condition](reports/figures/gpu_final_recall.png)

## Install (Windows PowerShell; Linux/macOS equivalents work too)
```powershell
git clone https://github.com/hammadfarooq-ai/opir-guardrail-obfuscation-eval.git
cd opir-guardrail-obfuscation-eval
py -3.13 -m venv .venv                 # Python >= 3.10
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; use a CUDA build for GPU
pip install -e ".[dev]"
$env:HF_HOME = "$PWD\.cache\hf"        # project-scoped model cache (the large checkpoint is about 1.6 GB)
```
`requirements.lock.txt` records the versions used locally (CPU build); the Kaggle run used torch 2.11.0+cu128, transformers 5.16.1 and gliclass 0.1.21 (`results/gpu/requirements_kaggle_freeze.txt`).

## Reproduce
```powershell
python scripts/prepare_data.py                                          # fetch pinned public data, rebuild manifests
python -m pytest -q tests                                               # tests
python scripts/run_eval.py --stage smoke --models opir-edge             # tiny smoke test (dev prompts)
python scripts/run_eval.py --stage pilot --models opir-edge opir-large  # 10+10 pilot (dev prompts)
python scripts/run_eval.py --stage dev   --models opir-edge opir-large
python scripts/run_eval.py --stage final --models opir-edge opir-large  # add --device cuda:0 --batch-size 16 on a GPU
python scripts/make_metrics.py --stage final                            # metrics, tables, figures from saved predictions
python scripts/extra_analysis.py --run gpu --stage final                # round-trip fidelity, tokenizer stats, error overlap
python scripts/export_errors.py --run gpu --stage final                 # per-prompt CSV
python scripts/bench_latency.py --device cpu --tag cpu                  # latency; CPU and GPU kept in separate files
```
- Resume an interrupted run: run the same `run_eval.py` command again (predictions are cached in `results/cache` by input hash, model revision and settings; failed inputs are retried).
- Regenerate metrics and figures for the shipped GPU results without any model: `python scripts/make_metrics.py --stage final --run gpu`.
- The shipped final results come from the Kaggle GPU run (`results/gpu/`); local CPU pilot results are in `results/pilot/`. A local CPU final run was not performed.
- Kaggle: `python scripts/build_kaggle_package.py --username <kaggle-user>` builds a private dataset and notebook package under `kaggle/`; the notebook installs `gliclass`, runs the tests, verifies that the manifests are reproduced, then runs pilot, dev and final one model at a time.

## Tests
`python -m pytest -q tests`: 13 passed (transforms, round trips, bounded base64 decoding, label mapping and split isolation, metric computation, mocked adapter and resume). Last run at commit `677dc8c`; later commits change documentation and packaging only. In the Kaggle run the data-dependent split test was skipped (raw data was fetched after the test cell); the manifests were instead verified to reproduce byte for byte.

## Data and licensing
JBB-Behaviors (MIT, JailbreakBench Team) and XSTest (CC-BY-4.0, Röttger et al.) prompts are redistributed in `data/manifests/` and one results CSV with attribution; raw downloads are re-fetched by `scripts/prepare_data.py` at pinned revisions and are not committed. Model weights are not included. Details and citations: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The repository's own code is released under the MIT License ([LICENSE](LICENSE)); the datasets keep their own licenses.

## Limitations (see [docs/limitations.md](docs/limitations.md))
Small single-source final set; false positives measured against the dataset's benign labels (possible policy mismatch is untested); one obfuscation at a time; normalizer tailored to the tested synthetic transforms; possible training-data overlap; single-label outputs only (no threshold study); timings apply to the tested machines and the stated benchmark inputs; not a complete guardrail evaluation.
