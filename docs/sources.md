# Verified sources

Access date: 2026-10-26. All items were read from the public Hugging Face pages/APIs listed; nothing was inferred from names.

## Models

| | opir-multitask-large | Opir-edge |
|---|---|---|
| Repo | `knowledgator/opir-multitask-large-v1.0` | `knowledgator/opir-edge-v1.0` |
| URL | https://huggingface.co/knowledgator/opir-multitask-large-v1.0 | https://huggingface.co/knowledgator/opir-edge-v1.0 |
| Revision (HF API `sha`) | `69bb27407d66eab4797049d36ba75eef3579335b` | `467a0431f744522f92f4e37d7756309b4c12332a` |
| License | Apache-2.0 | Apache-2.0 |
| Gated / credentials | No (`gated: false`) | No (`gated: false`) |
| Architecture | GLiClass uni-encoder, DeBERTa-v3-large backbone, ~0.4B params, F32 | GLiClass uni-encoder, Ettin-encoder-32m (ModernBERT) backbone, 32.7M params, F32 |
| Max input length | 1024 tokens (card: training maximum) | 1024 tokens |
| Safety task | binary `["safe","unsafe"]`, `classification_type="single-label"` | same |
| Output | list of `{"label","score"}`; card: take `max(result, key=score)`; single-label returns only the winning label | same |
| Threshold | none for binary (top label). 0.5 is documented only for multi-label use / edge "routing" example | same (`review_threshold=0.50` routing example) |
| Custom remote code | **None**: `config.json` has `architectures: ["GLiClassModel"]`; the repo holds only config, tokenizer and `model.safetensors` (no `.py` files). Code comes from the PyPI package `gliclass` | None; repo has no `.py` files |
| Install | `pip install gliclass transformers` (gliclass 0.1.21 requires transformers>=5.0, torch>=2.0) | same |

Decision: `trust_remote_code` is never enabled. Models are loaded with `GLiClassModel.from_pretrained(repo, revision=<sha>)`.

Opir-edge is the official smaller sibling listed in the large model card and in the Hugging Face collection https://huggingface.co/collections/knowledgator/opir. Both use the same labels and output format, so results are comparable in interface, but they are different models trained on different data files (large: `gliclass_full_en.json`, 426,356 examples; edge: `gliclass_safety_en.json`, 213,809 examples, per the cards).

Paper referenced by both cards: arXiv:2605.29659 (not used to take any numbers).

### Disclosed training data (contamination note)
Cards list taxonomy-derived synthetic prompts, hard-negative mining, `safe_and_benign` contrast examples, responses from a Qwen3-4B model fine-tuned on Aegis2, LLM-as-judge annotation, portions of Aegis2 and WildGuardMix training subsets, and replay data from `knowledgator/gliclass-v3-logic-dataset`. JBB-Behaviors and XSTest are not listed, but several JBB harmful behaviours derive from AdvBench/HarmBench, so overlap with training data cannot be ruled out.

## Datasets

| | JailbreakBench JBB-Behaviors | XSTest |
|---|---|---|
| Repo | `JailbreakBench/JBB-Behaviors` | `Paul/XSTest` (author's repo; the `walledai/XSTest` copy is gated and was not used) |
| Revision | `886acc352a31533ffbcf4ef22c744658688086fc` | `f600c994b256f12867dfa5b3eb3d545a3e62f8b5` |
| License | MIT | CC-BY-4.0 |
| Files | `data/harmful-behaviors.csv`, `data/benign-behaviors.csv` | `xstest_prompts.csv` |
| Labels | harmful split = unsafe; benign split = benign (labels supplied by dataset authors; the benign set is designed to match harmful topics) | `label` column: `safe` -> benign, `unsafe` -> unsafe (hand-written by authors) |
| Use here | Final evaluation: 100 unsafe + 100 benign | Development: 30 unsafe + 30 benign sampled (seed 20261106) |

Cards: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors , https://huggingface.co/datasets/Paul/XSTest . Citations: Chao et al. 2024 (JailbreakBench, NeurIPS D&B); Röttger et al. 2024 (XSTest, NAACL).

Redistribution: both licenses allow redistribution with attribution; manifests (including prompt text) are stored in `data/manifests/`, raw downloads are re-fetched by `scripts/prepare_data.py` and not committed.

## Infrastructure
- Local: Windows 10, Intel i5-7200U (4 threads), 7.9 GB RAM, no NVIDIA GPU, Python 3.13, torch 2.14.1+cpu, transformers 5.20.2.
- GPU: free Kaggle quota, private notebook (see `reports/final_status.md` for what actually ran).
