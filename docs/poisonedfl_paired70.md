# PoisonedFL paired experiment through round 70

The new launcher runs one clean warmup through round 14, saves the complete
server/client state, and restores that same state for eight branches:

`strength = 0, 1, 5, 10` crossed with `SPC = off, on`.

The attack is the project's fixed output-logit PoisonedFL-inspired controller.
It is not a claim to reproduce the original parameter-upload PoisonedFL paper.
All branches use the same fixed direction, optimizer momentum, learning-rate
scheduler horizon of 70, public split, and random state at the warmup boundary.

From the project root on the training server:

```bash
bash ssh/run_poisonedfl_paired70.sh 11121 screen plan
bash ssh/run_poisonedfl_paired70.sh 11121 screen run
python tools/summarize_poisonedfl_pair.py --base 11121
```

Use `smoke` first to validate the eight-branch control flow:

```bash
bash ssh/run_poisonedfl_paired70.sh 11221 smoke run
```

Set `PAIR_CHECKPOINT` to reuse a compatible round-14 checkpoint. The runner
checks a configuration/data signature and rejects incompatible or partial
snapshots. It also refuses to overwrite an existing run ID.

Each branch writes `*.metrics.jsonl`. Besides accuracy and ensemble accuracy,
the rows include raw pre-filter aggregate shift, effective post-filter shift,
SPC flagged/retained client IDs, TP/FP/FN/TN counts, and malicious clients
excluded by SPC. A branch is complete only when its last row is round 70.

Interpret SPC using both shifts: a drop in effective shift with unchanged raw
shift means filtering removed attack uploads; a drop in both means the attack
itself produced less poisoning. The `strength=0` SPC branch measures benign
false exclusions and should be reported alongside attack branches.
