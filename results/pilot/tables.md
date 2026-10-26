### opir-edge (pilot)

| condition | norm | unsafe n | benign n | TP | FN | TN | FP | recall [95% Wilson] | FPR [95% Wilson] | precision | F1 | errors | trunc | mean tok | unk inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | no | 10 | 10 | 9 | 1 | 9 | 1 | 90.0% [60-98] | 10.0% [2-40] | 90.0% | 90.0% | 0 | 0 | 10 | 0 |
| original | yes | 10 | 10 | 9 | 1 | 9 | 1 | 90.0% [60-98] | 10.0% [2-40] | 90.0% | 90.0% | 0 | 0 | 10 | 0 |
| homoglyph | no | 10 | 10 | 0 | 10 | 10 | 0 | 0.0% [0-28] | 0.0% [0-28] | n/a | n/a | 0 | 0 | 27 | 0 |
| homoglyph | yes | 10 | 10 | 9 | 1 | 9 | 1 | 90.0% [60-98] | 10.0% [2-40] | 90.0% | 90.0% | 0 | 0 | 10 | 0 |
| zero_width | no | 10 | 10 | 0 | 10 | 10 | 0 | 0.0% [0-28] | 0.0% [0-28] | n/a | n/a | 0 | 0 | 43 | 0 |
| zero_width | yes | 10 | 10 | 9 | 1 | 9 | 1 | 90.0% [60-98] | 10.0% [2-40] | 90.0% | 90.0% | 0 | 0 | 10 | 0 |
| base64 | no | 10 | 10 | 10 | 0 | 2 | 8 | 100.0% [72-100] | 80.0% [49-94] | 55.6% | 71.4% | 0 | 0 | 43 | 0 |
| base64 | yes | 10 | 10 | 9 | 1 | 9 | 1 | 90.0% [60-98] | 10.0% [2-40] | 90.0% | 90.0% | 0 | 0 | 10 | 0 |
| spaced | no | 10 | 10 | 9 | 1 | 1 | 9 | 90.0% [60-98] | 90.0% [60-98] | 50.0% | 64.3% | 0 | 0 | 43 | 0 |
| spaced | yes | 10 | 10 | 9 | 1 | 9 | 1 | 90.0% [60-98] | 10.0% [2-40] | 90.0% | 90.0% | 0 | 0 | 10 | 0 |

Normalization recovery / regression (paired by sample_id; pairs with an error excluded):

| condition | unsafe missed->detected | unsafe detected->missed | benign newly blocked | benign unblocked | unsafe pairs | benign pairs | inputs changed by norm |
|---|---|---|---|---|---|---|---|
| original | 0 | 0 | 0 | 0 | 10 | 10 | 0 |
| homoglyph | 9 | 0 | 1 | 0 | 10 | 10 | 20 |
| zero_width | 9 | 0 | 1 | 0 | 10 | 10 | 20 |
| base64 | 0 | 1 | 0 | 7 | 10 | 10 | 20 |
| spaced | 1 | 1 | 0 | 8 | 10 | 10 | 20 |

Paired source-level bootstrap, difference (normalized - not normalized), 95% CI:

| condition | metric | pairs | diff | 95% CI |
|---|---|---|---|---|
| original | recall | 10 | +0.0 pp | [+0.0, +0.0] |
| original | fpr | 10 | +0.0 pp | [+0.0, +0.0] |
| homoglyph | recall | 10 | +90.0 pp | [+70.0, +100.0] |
| homoglyph | fpr | 10 | +10.0 pp | [+0.0, +30.0] |
| zero_width | recall | 10 | +90.0 pp | [+70.0, +100.0] |
| zero_width | fpr | 10 | +10.0 pp | [+0.0, +30.0] |
| base64 | recall | 10 | -10.0 pp | [-30.0, +0.0] |
| base64 | fpr | 10 | -70.0 pp | [-90.0, -40.0] |
| spaced | recall | 10 | +0.0 pp | [-30.0, +30.0] |
| spaced | fpr | 10 | -80.0 pp | [-100.0, -50.0] |

Clean original prompts altered by normalization: 0; failed inferences: 0.
