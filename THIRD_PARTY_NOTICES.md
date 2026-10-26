# Third-party notices and data attribution

This repository redistributes prompt text from two public datasets (in `data/manifests/*.jsonl` and `results/gpu/final/original_text_predictions.csv`) and uses two public model checkpoints (not redistributed). The prompts include requests for harmful content (text only) and may be offensive. Sampling, whitespace stripping and ID assignment are the only changes made to the prompt text.

## JailbreakBench JBB-Behaviors (MIT)
- Source: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors, revision `886acc352a31533ffbcf4ef22c744658688086fc`.
- Citation: Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models", NeurIPS 2024 Datasets and Benchmarks Track.
- Used: the harmful split (100 prompts, labelled unsafe) and the benign split (100 prompts, labelled benign), unchanged apart from whitespace stripping.
- JBB states that some behaviours derive from AdvBench and HarmBench/TDC; this repository relies on the MIT license that JBB itself publishes and did not separately verify upstream terms.
- License text of the JBB dataset repository (reproduced as required by the MIT license):

```
MIT License

Copyright (c) 2023 JailbreakBench Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## XSTest (CC-BY-4.0)
- Source: https://huggingface.co/datasets/Paul/XSTest (the author's repository), revision `f600c994b256f12867dfa5b3eb3d545a3e62f8b5`; license field `cc-by-4.0`: https://creativecommons.org/licenses/by/4.0/
- Citation: Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models", NAACL 2024. Original repository: https://github.com/paul-rottger/xstest
- Used: a seeded sample of 30 `safe` and 30 `unsafe` prompts as the development split. Changes: whitespace stripping and sampling only. Labels are the authors'.
- Attribution: prompts (c) the XSTest authors, licensed under CC BY 4.0; no endorsement by the authors is implied.

## Models (not redistributed)
- `knowledgator/opir-multitask-large-v1.0` (revision `69bb27407d66eab4797049d36ba75eef3579335b`) and `knowledgator/opir-edge-v1.0` (revision `467a0431f744522f92f4e37d7756309b4c12332a`), Apache-2.0, loaded from the Hugging Face Hub at run time. Model predictions in `results/` are outputs of these models and carry no additional claim by their authors.

## Software
This project depends on `gliclass` (Apache-2.0), `transformers`, `torch`, `matplotlib` and `numpy` under their own licenses; none of their code is copied into this repository.

## License of this repository's own code
The code authored for this repository is licensed under the MIT License (see `LICENSE`, copyright Hammad Farooq). The MIT license of the JBB data above is separate and covers only that dataset.
