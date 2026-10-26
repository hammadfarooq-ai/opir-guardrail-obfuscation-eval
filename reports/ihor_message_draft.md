Hi Ihor,

Thank you for pointing me to the Opir checkpoints. As promised, I ran a small exploratory check of `opir-multitask-large-v1.0` and `opir-edge-v1.0` (binary safe/unsafe, single-label mode as in the model cards) on 100 unsafe + 100 benign prompts from JailbreakBench JBB-Behaviors, each with homoglyph, zero-width, base64 and spaced-letter variants, with and without a normalisation step (NFKC, zero-width removal, a small confusables map, strict one-layer base64 decoding, conservative spaced-letter joining). I also used 30+30 XSTest prompts as a separate dev set; no thresholds were tuned.

Two things stood out:
- Without normalisation, zero-width characters took recall from 97/100 to 1/100 (large) and 8/100 (edge), and homoglyphs to 68/100 (large) and 2/100 (edge). The large model's tokenizer emits [UNK] for the zero-width characters (all 200 such inputs); the edge tokenizer emitted none. With normalisation, recall returned to 97/100 for both, but the false-positive rate on the JBB benign prompts also returned to its unobfuscated level, which is high on this deliberately borderline set (54/100 large, 71/100 edge; XSTest safe prompts: 0/30 and 4/30).
- Base64 and spaced-letter text were mostly flagged as unsafe, benign prompts included (e.g. 93/100 benign flagged for large on spaced text without normalisation), so I would not read that as detection of the content.

Main caveat: this is a small, single-source set with one obfuscation at a time, and my normaliser is written for exactly these transforms, so the recovery is partly by construction. Results come from a free Kaggle T4 and a laptop CPU, and I can't rule out training-data overlap.

Code, configs and numbers: [repository link]. I'd be glad to hear whether the high false-positive rate on benign borderline prompts matches what you'd expect, and happy to try other settings you'd suggest.

Best regards,
Hammad
