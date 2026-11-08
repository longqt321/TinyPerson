# Error analysis — val

## Provenance

- Run: `/__modal/volumes/vo-1mSI8DTRRmOJaOvJIUm2al/gauss_cycle_1/20261101-215526-6fceb3b-seed42`
- Checkpoint: `/__modal/volumes/vo-1mSI8DTRRmOJaOvJIUm2al/gauss_cycle_1/20261101-215526-6fceb3b-seed42/training/weights/best.pt`
- Checkpoint SHA-256: `0cc0abcb9eeeaffa86e84afd59bf7fd021c80da5cda337ab16a8bf5ce019b86b`
- Dataset split: `val`
- Annotations SHA-256: `a33a2bcca802d2efac0d7b766c85c044f2421e8eea6c3ce469919c2c66119b67`
- Manifest SHA-256: `688e90348151d9801be2f96362865290c8dd83edd59f8d781696eda6fd09e718`
- Predictions SHA-256: `ba619d8c99418737f9d18c4382dfd994fd5cd1b8d7d5bd29b41a47bfa38d8838`
- Analysis config SHA-256: `1d2dabb5dfba11a76f01b94a29c9dcc5e6df67099cc2d596924027ebf019d5b8`
- Implementation: `tinydet-error-analysis-v1`

## Validity

- Status: **VALID**
- Expected images: 84
- Analyzed images: 84
- Dense images excluded by protocol: 0
- Predictions source: `/__modal/volumes/vo-1mSI8DTRRmOJaOvJIUm2al/gauss_cycle_1/20261101-215526-6fceb3b-seed42/analysis-val/predictions.json`

### Integrity findings

- `duplicate_annotation`: Image bb_V0031_I0000520.jpg: duplicate label 0 box [670.975451, 634.407596, 677.91498, 650.517216]
- `duplicate_annotation`: Image bb_V0031_I0000520.jpg: duplicate label 0 box [687.223861, 640.110284, 693.674318, 651.319274]
- `predictions_truncated`: Image baidu_P000_14.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image baidu_P000_16.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image baidu_P000_4.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0000000.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0000120.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0000240.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0000480.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0000600.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0000720.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0001080.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0001320.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0001720.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0001840.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0001960.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0002160.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0002240.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0002480.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0002600.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0002840.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0002880.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0003560.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0003680.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0004280.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0004320.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0004920.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0005160.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0005280.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0005520.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0005640.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0005880.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0006000.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0014_I0006040.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0000760.jpg: 209 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0000960.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0002200.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0002280.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0002320.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0002360.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0003200.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0005520.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0005640.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0007080.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0007240.jpg: 257 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bb_V0022_I0007880.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing000_2009040913572766058.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing000_48.jpg: 231 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing_P000_2008512103913872_2.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing_P000_367362_174157147172_2.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing_P000_5518b70991785.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing_P000_c5171330911.jpg: 293 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image bing_P000_timg (2).jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0005_I0001280.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0005_I0001480.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0005_I0001720.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0005_I0002200.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0005_I0003120.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0000000.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0000680.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0001960.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0002300.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0002500.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0003000.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0003160.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0004160.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0004200.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image chitube_V0008_I0004840.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image google_P000_10.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image google_P000_2.jpg: 239 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image google_P000_ItyC7NXlLRVZwM.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image google_P000_m4P9p77NH6rGQM.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image movie_V0000_I1.jpg: 300 predictions exceed max_detections; keeping 200
- `predictions_truncated`: Image movie_V0000_I5.jpg: 300 predictions exceed max_detections; keeping 200

## Observed counts

| quantity | count | denominator | rate |
| --- | --- | --- | --- |
| positive ground truths | 3608 |  |  |
| matched ground truths | 1640 | 3608 | 0.454545 |
| missed ground truths | 1968 | 3608 | 0.545455 |
| predictions | 15958 |  |  |
| true positives | 1640 |  |  |
| false positives | 14318 | 15958 | 0.10277 |
| duplicates | 1573 | 14318 | 0.109862 |
| localization errors | 4078 | 14318 | 0.284816 |
| background false positives | 8667 | 14318 | 0.605322 |
| neutral predictions | 0 |  |  |

These categories are exclusive for valid person predictions: each prediction is TP, FP, or neutral, and each positive ground truth is matched or missed. Precision here uses the predictions retained after the configured confidence threshold.

## Recall by object size

| diagnostic size bin | ground truths | matched | missed | recall | miss rate |
| --- | --- | --- | --- | --- | --- |
| tiny3 | 1106 | 525 | 581 | 0.474684 | 0.525316 |
| tiny2 | 916 | 493 | 423 | 0.53821 | 0.46179 |
| tiny1 | 805 | 267 | 538 | 0.331677 | 0.668323 |
| small | 553 | 229 | 324 | 0.414105 | 0.585895 |
| reasonable | 228 | 126 | 102 | 0.552632 | 0.447368 |

Diagnostic size bins are half-open `[low, high)` in sqrt(pixel area). The AP cross-check uses the repository evaluator's own inclusive size ranges.

## Recall and false positives by image density

| density bin | images | ground truths | matched | missed | recall | false positives | FP/image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| very_crowded | 29 | 2989 | 1245 | 1744 | 0.416527 | 4555 | 157.069 |
| sparse | 22 | 116 | 69 | 47 | 0.594828 | 3829 | 174.045 |
| single | 7 | 7 | 4 | 3 | 0.571429 | 1336 | 190.857 |
| crowded | 21 | 496 | 322 | 174 | 0.649194 | 3878 | 184.667 |
| none | 5 | 0 | 0 | 0 | n/a | 720 | 144 |

## Border versus interior objects

| position | ground truths | matched | missed | recall |
| --- | --- | --- | --- | --- |
| interior | 3463 | 1578 | 1885 | 0.455674 |
| border | 145 | 62 | 83 | 0.427586 |

## Confidence operating points

| score threshold | predictions | TP | FP | recall | precision |
| --- | --- | --- | --- | --- | --- |
| 0.001 | 15958 | 1640 | 14318 | 0.454545 | 0.10277 |
| 0.005 | 13848 | 1635 | 12213 | 0.45316 | 0.118068 |
| 0.01 | 12478 | 1627 | 10851 | 0.450942 | 0.130389 |
| 0.02 | 10806 | 1605 | 9201 | 0.444845 | 0.148529 |
| 0.05 | 8605 | 1570 | 7035 | 0.435144 | 0.182452 |
| 0.1 | 6176 | 1463 | 4713 | 0.405488 | 0.236885 |
| 0.2 | 3176 | 1126 | 2050 | 0.312084 | 0.354534 |
| 0.5 | 234 | 183 | 51 | 0.0507206 | 0.782051 |

The cumulative points reuse the primary-threshold matching outcomes. They are not a replacement for a full precision-recall curve at each threshold.

## Diagnostic AP cross-check

Status: **mismatch** against the repository TinyPerson evaluator.

| bin:metric | diagnostic | reference | absolute difference |
| --- | --- | --- | --- |
| all:AP@0.25 | 0.372329 | 0.372154 | 0.000175401 |
| all:AP@0.50 | 0.233633 | 0.233484 | 0.000148927 |
| all:AP@0.75 | 0.0227865 | 0.022771 | 1.55855e-05 |
| tiny:AP@0.25 | 0.379288 | 0.37898 | 0.000307621 |
| tiny:AP@0.50 | 0.233786 | 0.23382 | 3.32968e-05 |
| tiny:AP@0.75 | 0.0128103 | 0.0128055 | 4.84275e-06 |
| tiny1:AP@0.25 | 0.327095 | 0.326927 | 0.000168596 |
| tiny1:AP@0.50 | 0.149104 | 0.149122 | 1.7818e-05 |
| tiny1:AP@0.75 | 0.00217751 | 0.00217591 | 1.60116e-06 |
| tiny2:AP@0.25 | 0.444317 | 0.444145 | 0.000171703 |
| tiny2:AP@0.50 | 0.293667 | 0.293439 | 0.00022868 |
| tiny2:AP@0.75 | 0.0173031 | 0.0172805 | 2.26405e-05 |
| tiny3:AP@0.25 | 0.372943 | 0.372751 | 0.000192621 |
| tiny3:AP@0.50 | 0.256555 | 0.256241 | 0.000313631 |
| tiny3:AP@0.75 | 0.0233295 | 0.0233597 | 3.0179e-05 |
| small:AP@0.25 | 0.329441 | 0.329648 | 0.00020632 |
| small:AP@0.50 | 0.202193 | 0.202376 | 0.000183582 |
| small:AP@0.75 | 0.0159593 | 0.0160172 | 5.79353e-05 |
| reasonable:AP@0.25 | 0.446982 | 0.447117 | 0.000135003 |
| reasonable:AP@0.50 | 0.317066 | 0.316972 | 9.41653e-05 |
| reasonable:AP@0.75 | 0.0649787 | 0.0650455 | 6.68195e-05 |

## Investigation priorities

Ranking is by observed error count, then rate. It is a triage order, not causal attribution.

| category | group | count | denominator | rate | metric affected |
| --- | --- | --- | --- | --- | --- |
| false_positive | background | 8667 | 14318 | 0.605322 | precision |
| false_positive | localization | 4078 | 14318 | 0.284816 | precision |
| miss | density:very_crowded | 1744 | 2989 | 0.583473 | recall |
| false_positive | duplicate | 1573 | 14318 | 0.109862 | precision |
| miss | size_bin:tiny3 | 581 | 1106 | 0.525316 | recall |
| miss | size_bin:tiny1 | 538 | 805 | 0.668323 | recall |
| miss | size_bin:tiny2 | 423 | 916 | 0.46179 | recall |
| miss | size_bin:small | 324 | 553 | 0.585895 | recall |
| miss | density:crowded | 174 | 496 | 0.350806 | recall |
| miss | size_bin:reasonable | 102 | 228 | 0.447368 | recall |
| miss | density:sparse | 47 | 116 | 0.405172 | recall |
| miss | density:single | 3 | 7 | 0.428571 | recall |

## Hypotheses and follow-up

The tables above are measured facts. They do not identify a root cause. Candidate hypotheses must be tested with a controlled change and the same analysis pipeline.
- HYPOTHESIS: the `background` group may deserve investigation because it contains 8667 observed errors; test whether a targeted intervention changes this group without degrading the paired control group.
- NEXT: rerun this analysis on a paired checkpoint with identical dataset, split, score threshold, and matching configuration; compare sample-level regressions and corrections.

## Limitations

- Diagnostic AP is derived from sample records; the repository TinyPerson evaluator remains the metric source of truth.
- The derivative validation split is not the official test set.
- The TinyPerson evaluator is a repository implementation, not a vendored reference evaluator; miss rate still requires the reference evaluator.
- Associations in this report are descriptive, not causal.
- The prediction artifact carries no inference settings; imgsz, confidence, batch, and device are not independently verified.
