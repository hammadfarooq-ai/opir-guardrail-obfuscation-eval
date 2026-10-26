# opir-guardrail-eval

Independent, exploratory evaluation of two public Knowledgator Opir safety classifiers (`knowledgator/opir-multitask-large-v1.0`, `knowledgator/opir-edge-v1.0`) on public unsafe/benign prompts with common obfuscations (homoglyphs, zero-width characters, base64, spaced-out letters), with and without a documented normalization step. **Not an official Knowledgator benchmark.**

## Question
Do simple obfuscations reduce detection of unsafe prompts, does deterministic normalization recover it, and what does normalization do to false positives on benign prompts?

## What was actually run
- Final set: JailbreakBench JBB-Behaviors, **100 unsafe + 100 benign** prompts (MIT). Each prompt in 5 variants (original, homoglyph, zero_width, base64, spaced) x {no normalization, normalization}.
- Dev set: XSTest, 30 safe + 30 unsafe (CC-BY-4.0). Smoke (2+2) and pilot (10+10) subsets come from the dev set.
- Models (verified, public, ungated, Apache-2.0, no remote code): large rev `69bb27407d66`, edge rev `467a0431f744`. fp32, batch size 16, Tesla T4 on a private free Kaggle notebook (full run), plus a local CPU pilot and CPU latency benchmark (Intel i5-7200U, 7.9 GB RAM).
- 10 conditions per model (5 input forms x 2 normalization settings), 2000 rows / 1002 unique inputs per model, 0 inference errors and 0 truncated inputs in the final run; tests and executed commands are listed in `reports/final_status.md`.

## Main measured findings (final split, details in `reports/evaluation_report.md`)
| | recall original | FPR original (JBB benign) | recall, no norm: homoglyph / zero-width | recall with norm (all variants) |
|---|---|---|---|---|
| opir-large | 97/100 | 54/100 | 68/100 / 1/100 | 97/100 |
| opir-edge | 97/100 | 71/100 | 2/100 / 8/100 | 97/100 |

- Normalization is a tradeoff, not an unconditional improvement: on the four known synthetic transforms it restored the original-text behaviour, including the original false-positive level. That follows from a text-processing property (`normalize(transform(x))` reproduced the original prompt exactly for 398/400 transformed prompts, measured without any model); it is not evidence of robustness against unseen obfuscations.
- Base64 and spaced text were mostly flagged unsafe even when benign (e.g. large, spaced, no norm: 95/100 recall but 93/100 benign flagged), so that is not meaningful detection.
- The FPR figures are the observed flags on the JBB benign split under the dataset's labels, not a traffic estimate; whether some flags reflect a stricter policy than the dataset's is an untested interpretation (XSTest safe dev prompts, a different dataset: 0/30 large, 4/30 edge).
- Tokenizer observation (descriptive): the large model's tokenizer emitted `[UNK]` on the zero-width inputs (200/200); the edge tokenizer emitted none. This does not establish the cause of the prediction changes.

Code frozen at git tag `frozen-settings-v1` (commit `30d684f`). The pilot, dev and final stages ran automatically in one Kaggle notebook sequence, without a separate manual inspection gate; no setting was changed using final results. Audit of the headline numbers: `reports/evaluation_report.md` section 2.1.

## Install (Windows PowerShell)
```powershell
cd D:\Projects\opir-guardrail-eval
py -3.13 -m venv .venv            # any Python >= 3.10 works
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; use a CUDA build for GPU
pip install -e ".[dev]"            # or: pip install -r requirements.lock.txt
$env:HF_HOME = "$PWD\.cache\hf"    # project-scoped model cache
```

## Commands
```powershell
python scripts/prepare_data.py                                   # fetch pinned public data, write data/manifests
python -m pytest -q tests                                        # tests
python scripts/run_eval.py --stage smoke --models opir-edge      # tiny smoke test (dev prompts)
python scripts/run_eval.py --stage pilot --models opir-edge opir-large   # 10+10 pilot (dev prompts)
python scripts/run_eval.py --stage dev   --models opir-edge opir-large
python scripts/run_eval.py --stage final --models opir-edge opir-large   # add --device cuda:0 --batch-size 16 on GPU
# resume an interrupted run: run the same command again (cache in results/cache, errors are retried)
python scripts/make_metrics.py --stage final                     # metrics, tables, figures from saved predictions
python scripts/extra_analysis.py --run gpu --stage final         # normalization fidelity, tokenizer, error overlap
python scripts/bench_latency.py --device cpu --tag cpu           # latency (CPU and GPU kept in separate files)
```
GPU results live in `results/gpu/` (use `--run gpu` with `make_metrics.py`); local CPU pilot results in `results/pilot/`. The final stage was only run on Kaggle (a CPU final run of ~1 h for the large model is feasible locally with the same command).

### Kaggle (private notebook)
`python scripts/build_kaggle_package.py --username <kaggle-user>` builds `kaggle/dataset` and `kaggle/kernel`. Then `kaggle datasets create -p kaggle/dataset --dir-mode zip` and `kaggle kernels push -p kaggle/kernel` (private, GPU and Internet enabled). The notebook installs `gliclass`, runs tests, verifies the manifests are reproduced, then runs pilot, dev and final one model at a time.

## Layout
`src/opir_guardrail_eval/` transforms, normalize, data, models (adapter), runner, metrics; `scripts/`; `tests/`; `data/manifests/` frozen splits; `results/` predictions, metrics, run metadata, logs; `reports/`; `docs/` (sources, methodology, limitations).

## Data and licensing
JBB-Behaviors (MIT) and XSTest (CC-BY-4.0, Röttger et al.) manifests, including prompt text, are redistributed with attribution; raw downloads are fetched by `scripts/prepare_data.py` at pinned revisions and not committed. Prompts include offensive/harmful requests (text only). New code in this repository has no license file yet; choose one before publishing. Model weights are not included.

## Limitations (see `docs/limitations.md`)
Small single-source set; false positives measured against the dataset's benign labels (possible policy mismatch is untested); one obfuscation at a time; normalizer tailored to the tested synthetic transforms (2/400 round trips inexact; no evidence about unseen attacks); possible training-data overlap; single-label outputs only (no threshold study); timings are for the tested machines and short prompts only; not a complete guardrail evaluation.
