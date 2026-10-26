### opir-edge (final)

| condition | norm | unsafe n | benign n | TP | FN | TN | FP | recall [95% Wilson] | FPR [95% Wilson] | precision | F1 | errors | trunc | mean tok | unk inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | no | 100 | 100 | 97 | 3 | 29 | 71 | 97.0% [92-99] | 71.0% [61-79] | 57.7% | 72.4% | 0 | 0 | 15 | 0 |
| original | yes | 100 | 100 | 97 | 3 | 29 | 71 | 97.0% [92-99] | 71.0% [61-79] | 57.7% | 72.4% | 0 | 0 | 15 | 0 |
| homoglyph | no | 100 | 100 | 2 | 98 | 100 | 0 | 2.0% [1-7] | 0.0% [0-4] | 100.0% | 3.9% | 0 | 0 | 49 | 0 |
| homoglyph | yes | 100 | 100 | 97 | 3 | 29 | 71 | 97.0% [92-99] | 71.0% [61-79] | 57.7% | 72.4% | 0 | 0 | 15 | 0 |
| zero_width | no | 100 | 100 | 8 | 92 | 89 | 11 | 8.0% [4-15] | 11.0% [6-19] | 42.1% | 13.4% | 0 | 0 | 83 | 0 |
| zero_width | yes | 100 | 100 | 97 | 3 | 29 | 71 | 97.0% [92-99] | 71.0% [61-79] | 57.7% | 72.4% | 0 | 0 | 15 | 0 |
| base64 | no | 100 | 100 | 91 | 9 | 5 | 95 | 91.0% [84-95] | 95.0% [89-98] | 48.9% | 63.6% | 0 | 0 | 77 | 0 |
| base64 | yes | 100 | 100 | 97 | 3 | 29 | 71 | 97.0% [92-99] | 71.0% [61-79] | 57.7% | 72.4% | 0 | 0 | 15 | 0 |
| spaced | no | 100 | 100 | 100 | 0 | 0 | 100 | 100.0% [96-100] | 100.0% [96-100] | 50.0% | 66.7% | 0 | 0 | 80 | 0 |
| spaced | yes | 100 | 100 | 97 | 3 | 29 | 71 | 97.0% [92-99] | 71.0% [61-79] | 57.7% | 72.4% | 0 | 0 | 15 | 0 |

Normalization recovery / regression (paired by sample_id; pairs with an error excluded):

| condition | unsafe missed->detected | unsafe detected->missed | benign newly blocked | benign unblocked | unsafe pairs | benign pairs | inputs changed by norm |
|---|---|---|---|---|---|---|---|
| original | 0 | 0 | 0 | 0 | 100 | 100 | 0 |
| homoglyph | 95 | 0 | 71 | 0 | 100 | 100 | 200 |
| zero_width | 89 | 0 | 61 | 1 | 100 | 100 | 200 |
| base64 | 9 | 3 | 3 | 27 | 100 | 100 | 200 |
| spaced | 0 | 3 | 0 | 29 | 100 | 100 | 200 |

Paired source-level bootstrap, difference (normalized - not normalized), 95% CI:

| condition | metric | pairs | diff | 95% CI |
|---|---|---|---|---|
| original | recall | 100 | +0.0 pp | [+0.0, +0.0] |
| original | fpr | 100 | +0.0 pp | [+0.0, +0.0] |
| homoglyph | recall | 100 | +95.0 pp | [+90.0, +99.0] |
| homoglyph | fpr | 100 | +71.0 pp | [+62.0, +80.0] |
| zero_width | recall | 100 | +89.0 pp | [+82.0, +95.0] |
| zero_width | fpr | 100 | +60.0 pp | [+50.0, +70.0] |
| base64 | recall | 100 | +6.0 pp | [+0.0, +13.0] |
| base64 | fpr | 100 | -24.0 pp | [-34.0, -14.0] |
| spaced | recall | 100 | -3.0 pp | [-7.0, +0.0] |
| spaced | fpr | 100 | -29.0 pp | [-38.0, -20.0] |

Clean original prompts altered by normalization: 0; failed inferences: 0.

### opir-large (final)

| condition | norm | unsafe n | benign n | TP | FN | TN | FP | recall [95% Wilson] | FPR [95% Wilson] | precision | F1 | errors | trunc | mean tok | unk inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | no | 100 | 100 | 97 | 3 | 46 | 54 | 97.0% [92-99] | 54.0% [44-63] | 64.2% | 77.3% | 0 | 0 | 14 | 0 |
| original | yes | 100 | 100 | 97 | 3 | 46 | 54 | 97.0% [92-99] | 54.0% [44-63] | 64.2% | 77.3% | 0 | 0 | 14 | 0 |
| homoglyph | no | 100 | 100 | 68 | 32 | 67 | 33 | 68.0% [58-76] | 33.0% [25-43] | 67.3% | 67.7% | 0 | 0 | 47 | 0 |
| homoglyph | yes | 100 | 100 | 97 | 3 | 46 | 54 | 97.0% [92-99] | 54.0% [44-63] | 64.2% | 77.3% | 0 | 0 | 14 | 0 |
| zero_width | no | 100 | 100 | 1 | 99 | 95 | 5 | 1.0% [0-5] | 5.0% [2-11] | 16.7% | 1.9% | 0 | 0 | 48 | 200 |
| zero_width | yes | 100 | 100 | 97 | 3 | 46 | 54 | 97.0% [92-99] | 54.0% [44-63] | 64.2% | 77.3% | 0 | 0 | 14 | 0 |
| base64 | no | 100 | 100 | 81 | 19 | 16 | 84 | 81.0% [72-87] | 84.0% [76-90] | 49.1% | 61.1% | 0 | 0 | 76 | 0 |
| base64 | yes | 100 | 100 | 97 | 3 | 46 | 54 | 97.0% [92-99] | 54.0% [44-63] | 64.2% | 77.3% | 0 | 0 | 14 | 0 |
| spaced | no | 100 | 100 | 95 | 5 | 7 | 93 | 95.0% [89-98] | 93.0% [86-97] | 50.5% | 66.0% | 0 | 0 | 68 | 0 |
| spaced | yes | 100 | 100 | 97 | 3 | 46 | 54 | 97.0% [92-99] | 54.0% [44-63] | 64.2% | 77.3% | 0 | 0 | 14 | 0 |

Normalization recovery / regression (paired by sample_id; pairs with an error excluded):

| condition | unsafe missed->detected | unsafe detected->missed | benign newly blocked | benign unblocked | unsafe pairs | benign pairs | inputs changed by norm |
|---|---|---|---|---|---|---|---|
| original | 0 | 0 | 0 | 0 | 100 | 100 | 0 |
| homoglyph | 29 | 0 | 28 | 7 | 100 | 100 | 200 |
| zero_width | 97 | 1 | 52 | 3 | 100 | 100 | 200 |
| base64 | 18 | 2 | 9 | 39 | 100 | 100 | 200 |
| spaced | 5 | 3 | 5 | 44 | 100 | 100 | 200 |

Paired source-level bootstrap, difference (normalized - not normalized), 95% CI:

| condition | metric | pairs | diff | 95% CI |
|---|---|---|---|---|
| original | recall | 100 | +0.0 pp | [+0.0, +0.0] |
| original | fpr | 100 | +0.0 pp | [+0.0, +0.0] |
| homoglyph | recall | 100 | +29.0 pp | [+20.0, +39.0] |
| homoglyph | fpr | 100 | +21.0 pp | [+10.0, +32.0] |
| zero_width | recall | 100 | +96.0 pp | [+91.0, +100.0] |
| zero_width | fpr | 100 | +49.0 pp | [+37.0, +60.0] |
| base64 | recall | 100 | +16.0 pp | [+8.0, +24.0] |
| base64 | fpr | 100 | -30.0 pp | [-42.0, -18.0] |
| spaced | recall | 100 | +2.0 pp | [-3.0, +7.0] |
| spaced | fpr | 100 | -39.0 pp | [-50.0, -28.0] |

Clean original prompts altered by normalization: 0; failed inferences: 0.
