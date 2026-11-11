# SPC threshold sensitivity: k=2,3,4 (plus k=5 reference)

Offline classification of fixed historical trajectories only. No code changes or retraining. These results do not predict final model accuracy.

Sources: logs/12104_cpa_plain_spc.log (attack), logs/12102_clear_spc.log (clean), and corresponding analysis.json files. Rounds 15-70: 56 rounds, 504 malicious uploads and 1176 honest uploads in the attack run, 1680 honest uploads in the clean run. Counts are client-round events, not independent clients or replicated experiments.

Cutoff = core median + k * core sample standard deviation. Exclude strictly higher scores. densest_half uses the original logged core statistics. lowest_half takes the 15 lowest Soft-JS scores each round. Malicious labels are used for evaluation only.

| Core | k | Detection rate | Misses /504 | Honest exclusions: attack /1176 | Honest exclusions: clean /1680 | Leak rounds |
|---|---:|---:|---:|---:|---:|---|
| densest_half | 2 | 94.64% | 27 | 145 (12.33%) | 483 (28.75%) | 17,44,50 |
| densest_half | 3 | 94.64% | 27 | 67 (5.70%) | 340 (20.24%) | 17,44,50 |
| densest_half | 4 | 94.64% | 27 | 25 (2.13%) | 230 (13.69%) | 17,44,50 |
| densest_half | 5 | 92.86% | 36 | 16 (1.36%) | 148 (8.81%) | 17,37,44,50 |
| lowest_half | 2 | 100.00% | 0 | 191 (16.24%) | 606 (36.07%) | none |
| lowest_half | 3 | 100.00% | 0 | 70 (5.95%) | 361 (21.49%) | none |
| lowest_half | 4 | 94.64% | 27 | 23 (1.96%) | 199 (11.85%) | 15,44,50 |
| lowest_half | 5 | 92.86% | 36 | 10 (0.85%) | 113 (6.73%) | 15,17,44,50 |

## Rounding uncertainty

Soft-JS is printed to 4 decimals; original core statistics to 6 decimals. Counts other than the actual original k=5 configuration are point estimates. Conservative rounding ranges follow; these are NOT statistical confidence intervals. All malicious miss counts remain unchanged within these rounding bounds.

| Core | k | Attack honest exclusion range | Clean honest exclusion range |
|---|---:|---:|---:|
| densest_half | 2 | 144-145 | 483-484 |
| densest_half | 3 | 67-67 | 340-342 |
| densest_half | 4 | 24-25 | 230-231 |
| densest_half | 5 | 16-16 | 148-148 |
| lowest_half | 2 | 191-195 | 603-614 |
| lowest_half | 3 | 70-72 | 356-364 |
| lowest_half | 4 | 22-23 | 195-201 |
| lowest_half | 5 | 9-10 | 109-116 |

Score error bound: +/-0.00005; original core median and std errors: each +/-0.0000005. For lowest_half, each sorted score error is bounded by +/-0.00005; sample std error is conservatively bounded by sqrt(15/14)*0.00005. Propagate these to cutoff bounds and compare score intervals.

## Interpretation

- densest_half at k=2,3,4 misses all nine attackers in rounds 17,44,50. Nine of the 15 core members are malicious in these rounds, placing the median at their score. Positive k does not resolve this structural issue.
- lowest_half at k=2 and k=3 detects all malicious uploads on these historical trajectories. Clean false positive rates are approximately 36.07% and 21.49%; k=2 gives no extra detection benefit here.
- lowest_half with k=3 is a candidate for separate validation, not an established optimum or complete protection. Results do not establish performance against HIPS. Previously examined attack logs are not an independent validation set.
- Changing screening changes subsequent server models and client predictions. Re-training can change scores, misses, false positives and model accuracy.
