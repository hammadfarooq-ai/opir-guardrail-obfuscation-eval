# Limitations

- **Small, single-source final set**: 100 + 100 prompts from JBB-Behaviors only; Wilson intervals are wide (for n=100 roughly +/-10 points near 50%). Results are exploratory, not a benchmark.
- **Short, direct prompts**: JBB behaviours are one-sentence requests, not jailbreak templates, multi-turn chats or long documents. Results may not transfer to those.
- **Possible contamination**: training sources are only partially disclosed; JBB harmful items derive from AdvBench/HarmBench, which other safety datasets reuse. Overlap cannot be ruled out.
- **Benign set is synthetic and policy-dependent**: JBB benign behaviours were produced to mirror harmful topics; a few are borderline and the dataset authors' labels may differ from a given deployer's policy. XSTest (dev) is hand-written and keyword-heavy. No real-user benign traffic was tested.
- **Obfuscations are simple and deterministic**: one transform at a time; no combined or adaptive attacks; no embedded base64 segments inside longer text (the normalizer decodes whole-text base64 only); homoglyph choice is a fixed rule, so an adversary could pick others.
- **Normalization was designed with these transforms in mind**: it inverts exactly the documented obfuscations, so its recovery is partly by construction. It should not be read as robustness to unseen obfuscations. The confusables map is deliberately small. A mixed-script word in genuine multilingual text could be altered.
- **Base64 / spaced tests change the task**: the readable meaning is absent (base64) or fragmented (spaced). A "detection" there is not evidence of understanding; a flag on benign base64 text shows the model reacting to gibberish. Both detection and FPR must be read together.
- **Binary output only**: single-label mode exposes only the top label and its score, so threshold/ROC analysis was not possible; scores are not treated as calibrated probabilities.
- **No tuning**: no threshold or prompt tuning was done; a deployer might recalibrate or fine-tune (Ihor noted application-specific fine-tuning may be needed).
- **Hardware/timing**: CPU timings are from a 2-core/4-thread laptop CPU and GPU timings from whatever free Kaggle GPU was assigned; neither represents production serving. Free-tier GPU timings can vary run to run.
- **Scope**: this does not measure PII protection, citation verification, LLM fallback, response-side safety, multilingual input, or a complete production guardrail.
