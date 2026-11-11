# PoisonedFL strength 10: SPC filtering test

Run `ssh/run_poisonedfl_s10_spc.sh BASE_ID full|smoke plan|run`.
The default is `11501 full plan`; plan mode makes no changes.

Full mode requires the original `checkpoints/poisonedfl_pair_11111_round14.pt`.
Override `PAIR_CHECKPOINT` only with the compatible shared clean round-14 state.
The runner validates the configuration/data signature before restoring it.

| ID | Condition | Rounds |
|---|---|---|
|11501|Restore shared clean checkpoint|14|
|11502|No attack + SPC filtering|15–70|
|11503|Fixed PoisonedFL-inspired logit attack, strength 10 + SPC filtering|15–70|

Both branches restore the same models, optimizer, scheduler and random state.
All previous data/probes/LR settings remain unchanged: CIFAR-10/ResNet18,
30 clients (9 malicious in the attacked branch), seed 20260805, T=1, Mean,
70-round LR horizon, SPC core-sigma threshold 5, no adaptive strength.
The warmup keeps filtering off; both named SPC branches enable it in the runner.
The initial CLI `probe_core_sigma_filter=false` preserves the original warmup
signature and does not disable filtering in these two branches.

The new `poisonedfl_pair_spc_only` option defaults to false and is excluded from
the warmup signature. Existing four/eight-branch modes retain their behavior.
Only two new full training branches run. Existing no-filter references are:

- 11302 clean Mean: server 71.79%, ensemble 71.70%.
- 11305 strength-10 Mean: server 31.55%, ensemble 37.40%.

Measure defense recovery against 11305, remaining damage against the new clean
SPC branch, and clean filtering cost against 11302. Final metrics and the last
10 rounds should both be compared. `delta_vs_clear_spc_pp` (also exposed as
`delta_vs_clear_pp`) uses the new clean SPC branch as its same-round baseline.
It is not the old no-filter baseline. Report detection and false exclusion from
all 56 rows, plus actual excluded client IDs; audit flags alone are not removals.

Files: `logs/11501_11503_poisonedfl_paired.summary.txt`, per-branch `.log`, and
`11502_poisonedfl_pair_clear_spc.metrics.jsonl` /
`11503_poisonedfl_pair_strength10_spc.metrics.jsonl`.
Both branches must contain consecutive rounds 15–70 and exit 0.

Smoke mode uses 4 clients, 1 malicious, shared round 1 and two rounds per branch.
Use a separate base, e.g. `11601 smoke run`, and do not set the full checkpoint
environment variable when running smoke. No server experiments are auto-started.

Local validation: 70 unittest cases, 69 passed and 1 CUDA-only case skipped.
The new two-branch test exercises real synthetic training/filtering and checkpoint
reuse with a controlled audit fixture; existing four/eight-branch regressions pass.
Bash syntax, full/smoke plans, Python 3.10 syntax, and actual CLI boolean
conversion passed. No new GPU/data training has been run remotely.
