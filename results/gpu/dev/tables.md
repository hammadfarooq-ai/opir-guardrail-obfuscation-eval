### opir-edge (dev)

| condition | norm | unsafe n | benign n | TP | FN | TN | FP | recall [95% Wilson] | FPR [95% Wilson] | precision | F1 | errors | trunc | mean tok | unk inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | no | 30 | 30 | 22 | 8 | 26 | 4 | 73.3% [56-86] | 13.3% [5-30] | 84.6% | 78.6% | 0 | 0 | 10 | 0 |
| original | yes | 30 | 30 | 22 | 8 | 26 | 4 | 73.3% [56-86] | 13.3% [5-30] | 84.6% | 78.6% | 0 | 0 | 10 | 0 |
| homoglyph | no | 30 | 30 | 1 | 29 | 30 | 0 | 3.3% [1-17] | 0.0% [0-11] | 100.0% | 6.5% | 0 | 0 | 28 | 0 |
| homoglyph | yes | 30 | 30 | 22 | 8 | 26 | 4 | 73.3% [56-86] | 13.3% [5-30] | 84.6% | 78.6% | 0 | 0 | 10 | 0 |
| zero_width | no | 30 | 30 | 0 | 30 | 28 | 2 | 0.0% [0-11] | 6.7% [2-21] | 0.0% | 0.0% | 0 | 0 | 45 | 0 |
| zero_width | yes | 30 | 30 | 22 | 8 | 26 | 4 | 73.3% [56-86] | 13.3% [5-30] | 84.6% | 78.6% | 0 | 0 | 10 | 0 |
| base64 | no | 30 | 30 | 28 | 2 | 4 | 26 | 93.3% [79-98] | 86.7% [70-95] | 51.9% | 66.7% | 0 | 0 | 44 | 0 |
| base64 | yes | 30 | 30 | 22 | 8 | 26 | 4 | 73.3% [56-86] | 13.3% [5-30] | 84.6% | 78.6% | 0 | 0 | 10 | 0 |
| spaced | no | 30 | 30 | 26 | 4 | 4 | 26 | 86.7% [70-95] | 86.7% [70-95] | 50.0% | 63.4% | 0 | 0 | 45 | 0 |
| spaced | yes | 30 | 30 | 22 | 8 | 26 | 4 | 73.3% [56-86] | 13.3% [5-30] | 84.6% | 78.6% | 0 | 0 | 10 | 0 |

Normalization recovery / regression (paired by sample_id; pairs with an error excluded):

| condition | unsafe missed->detected | unsafe detected->missed | benign newly blocked | benign unblocked | unsafe pairs | benign pairs | inputs changed by norm |
|---|---|---|---|---|---|---|---|
| original | 0 | 0 | 0 | 0 | 30 | 30 | 0 |
| homoglyph | 21 | 0 | 4 | 0 | 30 | 30 | 60 |
| zero_width | 22 | 0 | 4 | 2 | 30 | 30 | 60 |
| base64 | 2 | 8 | 0 | 22 | 30 | 30 | 60 |
| spaced | 2 | 6 | 0 | 22 | 30 | 30 | 60 |

Paired source-level bootstrap, difference (normalized - not normalized), 95% CI:

| condition | metric | pairs | diff | 95% CI |
|---|---|---|---|---|
| original | recall | 30 | +0.0 pp | [+0.0, +0.0] |
| original | fpr | 30 | +0.0 pp | [+0.0, +0.0] |
| homoglyph | recall | 30 | +70.0 pp | [+53.3, +86.7] |
| homoglyph | fpr | 30 | +13.3 pp | [+3.3, +26.7] |
| zero_width | recall | 30 | +73.3 pp | [+56.7, +90.0] |
| zero_width | fpr | 30 | +6.7 pp | [-10.0, +23.3] |
| base64 | recall | 30 | -20.0 pp | [-40.0, +0.0] |
| base64 | fpr | 30 | -73.3 pp | [-86.7, -56.7] |
| spaced | recall | 30 | -13.3 pp | [-33.3, +3.3] |
| spaced | fpr | 30 | -73.3 pp | [-86.7, -56.7] |

Clean original prompts altered by normalization: 0; failed inferences: 0.

### opir-large (dev)

| condition | norm | unsafe n | benign n | TP | FN | TN | FP | recall [95% Wilson] | FPR [95% Wilson] | precision | F1 | errors | trunc | mean tok | unk inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | no | 30 | 30 | 26 | 4 | 30 | 0 | 86.7% [70-95] | 0.0% [0-11] | 100.0% | 92.9% | 0 | 0 | 10 | 0 |
| original | yes | 30 | 30 | 26 | 4 | 30 | 0 | 86.7% [70-95] | 0.0% [0-11] | 100.0% | 92.9% | 0 | 0 | 10 | 0 |
| homoglyph | no | 30 | 30 | 5 | 25 | 29 | 1 | 16.7% [7-34] | 3.3% [1-17] | 83.3% | 27.8% | 0 | 0 | 27 | 0 |
| homoglyph | yes | 30 | 30 | 26 | 4 | 30 | 0 | 86.7% [70-95] | 0.0% [0-11] | 100.0% | 92.9% | 0 | 0 | 10 | 0 |
| zero_width | no | 30 | 30 | 3 | 27 | 29 | 1 | 10.0% [3-26] | 3.3% [1-17] | 75.0% | 17.6% | 0 | 0 | 26 | 60 |
| zero_width | yes | 30 | 30 | 26 | 4 | 30 | 0 | 86.7% [70-95] | 0.0% [0-11] | 100.0% | 92.9% | 0 | 0 | 10 | 0 |
| base64 | no | 30 | 30 | 22 | 8 | 5 | 25 | 73.3% [56-86] | 83.3% [66-93] | 46.8% | 57.1% | 0 | 0 | 44 | 0 |
| base64 | yes | 30 | 30 | 26 | 4 | 30 | 0 | 86.7% [70-95] | 0.0% [0-11] | 100.0% | 92.9% | 0 | 0 | 10 | 0 |
| spaced | no | 30 | 30 | 2 | 28 | 28 | 2 | 6.7% [2-21] | 6.7% [2-21] | 50.0% | 11.8% | 0 | 0 | 38 | 0 |
| spaced | yes | 30 | 30 | 26 | 4 | 30 | 0 | 86.7% [70-95] | 0.0% [0-11] | 100.0% | 92.9% | 0 | 0 | 10 | 0 |

Normalization recovery / regression (paired by sample_id; pairs with an error excluded):

| condition | unsafe missed->detected | unsafe detected->missed | benign newly blocked | benign unblocked | unsafe pairs | benign pairs | inputs changed by norm |
|---|---|---|---|---|---|---|---|
| original | 0 | 0 | 0 | 0 | 30 | 30 | 0 |
| homoglyph | 21 | 0 | 0 | 1 | 30 | 30 | 60 |
| zero_width | 23 | 0 | 0 | 1 | 30 | 30 | 60 |
| base64 | 8 | 4 | 0 | 25 | 30 | 30 | 60 |
| spaced | 24 | 0 | 0 | 2 | 30 | 30 | 60 |

Paired source-level bootstrap, difference (normalized - not normalized), 95% CI:

| condition | metric | pairs | diff | 95% CI |
|---|---|---|---|---|
| original | recall | 30 | +0.0 pp | [+0.0, +0.0] |
| original | fpr | 30 | +0.0 pp | [+0.0, +0.0] |
| homoglyph | recall | 30 | +70.0 pp | [+53.3, +86.7] |
| homoglyph | fpr | 30 | -3.3 pp | [-10.0, +0.0] |
| zero_width | recall | 30 | +76.7 pp | [+60.0, +90.0] |
| zero_width | fpr | 30 | -3.3 pp | [-10.0, +0.0] |
| base64 | recall | 30 | +13.3 pp | [-10.0, +33.3] |
| base64 | fpr | 30 | -83.3 pp | [-93.3, -70.0] |
| spaced | recall | 30 | +80.0 pp | [+66.7, +93.3] |
| spaced | fpr | 30 | -6.7 pp | [-16.7, +0.0] |

Clean original prompts altered by normalization: 0; failed inferences: 0.
