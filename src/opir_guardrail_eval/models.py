"""Model adapters. Each adapter keeps its own label/score semantics and returns raw outputs."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

MODELS = {
    "opir-large": {"repo": "knowledgator/opir-multitask-large-v1.0",
                   "revision": "69bb27407d66eab4797049d36ba75eef3579335b"},
    "opir-edge": {"repo": "knowledgator/opir-edge-v1.0",
                  "revision": "467a0431f744522f92f4e37d7756309b4c12332a"},
}
LABELS = ["safe", "unsafe"]  # as documented in both model cards (single-label binary mode)
MAX_LENGTH = 1024  # documented training/max sequence length


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


@dataclass
class GLiClassAdapter:
    """Official quick-start interface: ZeroShotClassificationPipeline, single-label, ["safe","unsafe"].

    Decision rule (documented): the top label wins; no extra threshold or score transform is
    applied. The full pipeline output is preserved as ``raw``.
    """
    name: str
    device: str = "cpu"
    batch_size: int = 4
    seed: int = 0
    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)
    pipe: object = field(default=None, repr=False)
    load_seconds: float = 0.0

    @property
    def repo(self): return MODELS[self.name]["repo"]
    @property
    def revision(self): return MODELS[self.name]["revision"]

    def settings(self) -> dict:
        return {"repo": self.repo, "revision": self.revision, "labels": LABELS, "mode": "single-label",
                "max_length": MAX_LENGTH, "device": self.device, "dtype": "float32",
                "decision": "top label of single-label output"}

    def settings_key(self) -> str:
        s = self.settings()
        s.pop("device")  # cache is valid across devices up to numerical noise
        return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]

    def load(self):
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        self.model = GLiClassModel.from_pretrained(self.repo, revision=self.revision)  # no remote code
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        self.model.eval()
        self.pipe = ZeroShotClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer, classification_type="single-label",
            device=self.device, max_length=MAX_LENGTH, progress_bar=False)
        self.load_seconds = time.perf_counter() - t0
        return self

    def token_stats(self, text: str) -> dict:
        """Tokens of the text alone, plus whether the full classifier input (labels + text) is truncated."""
        tok = self.tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]
        unk = tok.unk_token_id
        full = self.pipe.pipe.prepare_input(text, LABELS, None, None)
        full_len = len(tok(full, truncation=False)["input_ids"])
        return {"n_tokens": len(ids), "n_unk": sum(i == unk for i in ids) if unk is not None else None,
                "full_input_tokens": full_len, "truncated": full_len > MAX_LENGTH}

    def predict(self, texts: list[str]) -> list[dict]:
        import torch
        with torch.inference_mode():
            out = self.pipe(texts, LABELS, batch_size=self.batch_size)
        return [self._parse(o) for o in out]

    @staticmethod
    def _parse(o) -> dict:
        """Single-label mode returns only the winning label with its score (documented usage:
        ``max(result, key=score)``). The runner-up score is not exposed, so none is inferred."""
        if not o:
            raise ValueError(f"empty pipeline output: {o}")
        top = max(o, key=lambda d: d["score"])
        if top["label"] not in LABELS:
            raise ValueError(f"unexpected label in output: {o}")
        return {"top_label": top["label"], "top_score": float(top["score"]),
                "pred_unsafe": top["label"] == "unsafe", "raw": o}
