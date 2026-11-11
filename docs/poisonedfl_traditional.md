# PoisonedFL-inspired logits strength 10 vs traditional defences

Run `bash ssh/run_poisonedfl_traditional.sh BASE_ID full|smoke plan|run`.
Default: `11701 full plan`. Planning is read-only.

## Six new branches

| ID | Condition | Rounds |
|---|---|---|
|11701|Restore the original shared clean Mean checkpoint|14|
|11702|No attack + TriMean|15–70|
|11703|Strength 10 + TriMean|15–70|
|11704|No attack + Multi-Krum|15–70|
|11705|Strength 10 + Multi-Krum|15–70|
|11706|No attack + AFA|15–70|
|11707|Strength 10 + AFA|15–70|

All six branches independently restore the same original
`checkpoints/poisonedfl_pair_11111_round14.pt`, including server/client models,
optimizer momentum, schedulers, scaler and RNG. Full mode requires this existing
checkpoint (or `PAIR_CHECKPOINT` pointing to an equivalent compatible state).
The original configuration/data signature is checked before each restoration.

The first 14 rounds are Mean warmup, exactly as in the existing Mean/SPC tests.
Traditional aggregation starts at round 15; this design does not claim to train
with robust aggregation from round 1. AFA's reliability/blocking history starts
fresh at round 15 in each branch and persists for that branch's 56 rounds.
Do not substitute earlier PCFDLA clean controls trained with a different warmup.

## Registered parameters

- CIFAR-10 / ResNet18, 30 clients, 9 malicious in attacks, seed 20260805.
- Fixed PoisonedFL-inspired output direction, strength 10, adaptive=false.
- Private data 47500, validation 2500, full public STL-10 unlabeled 100000.
- 70-round LR horizon, SGD momentum .9, weight decay .0001, batch 128, T=1.
- Same scale/replace probes and audit as previous groups; SPC filtering off.
- TriMean: coordinate-wise trim ratio .3, 9 values removed per tail at n=30.
- Multi-Krum: explicit f=9, 3 selected, 19 nearest distances, fallback=error.
  The clean branch also keeps f=9 rather than treating its actual malicious count
  of zero as a different aggregation configuration.
- AFA: xi=2, delta_xi=.5, prior_alpha=3, prior_beta=3, block_threshold=.95.

Existing algorithms in `byzantine/defences.py` are unchanged. The new paired
mode wraps the actual aggregator for diagnostics and delegates client IDs and
state. It does not replace TriMean/Krum/AFA by the Mean diagnostic aggregator.
Branch settings are installed only after the shared warmup signature has been
computed; they do not retroactively change the saved Mean warmup's config.

## Existing references (no repeat training)

| Group | Clean server | Strength-10 server | Clean ensemble | Strength-10 ensemble |
|---|---:|---:|---:|---:|
|Mean: 11302 / 11305|71.79%|31.55%|71.70%|37.40%|
|SPC: 11502 / 11503|71.14%|72.06%|71.57%|72.30%|

Compare each new attacked group with its own defence's clean group. JSONL
`delta_vs_clean_defence_pp` and its alias `delta_vs_clear_pp` are that same-round
server accuracy difference in percentage points. Negative means degradation.
They do not compare against Mean or another defence's clean group.

All phases have separate logs, each branch has `.metrics.jsonl`, and the final
summary is `logs/11701_11707_poisonedfl_paired.summary.txt` for the default base.
Report final and last-10-round accuracy. Aggregation logs expose Multi-Krum
malicious selections and AFA malicious retention/blocking; TriMean is a
coordinate-wise rule and has no single per-client retained set.

Probe flags remain audit-only in all new branches. `spc_excluded_ids` describes
SPC filtering, not which clients a traditional aggregator selected. Likewise,
`aggregate_shift_rms` uses the current branch's unpoisoned all-client Mean as
its reference and includes the robust rule's own aggregation change; it must
not be interpreted as the pure residual attack after robust aggregation.

Smoke mode uses 6 clients, 1 malicious, shared round 1 and two rounds per branch,
so Multi-Krum's f=1 / selected=3 is feasible. Use `11801 smoke plan|run` with
`PAIR_CHECKPOINT` unset so it does not try to restore the full 30-client state.
No server jobs are started automatically.

Validation: 73 unittest cases completed locally, 72 passed and 1 CUDA-only case
skipped. Tests cover all six real synthetic branches, shared checkpoint reuse,
separate same-defence clean baselines, observer equivalence to actual robust
aggregators, AFA state persistence/reset, and the existing four/eight/SPC-only
modes. Bash syntax, full/smoke plans, CLI boolean conversion and Python 3.10
syntax passed. No remote GPU experiment was launched during preparation.
