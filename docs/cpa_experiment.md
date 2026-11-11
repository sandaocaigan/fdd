# CPA review and experiment

## Provenance

CPA was present in the upstream repository's initial commit, dated
2024-10-11 15:01:33 +0200:

https://github.com/ZIB-IOL/FedDistill/commit/7481aeadef9eff6b4672213cb38a9334408ef555

The local project has no `.git` directory, so this does not establish when it
was copied into this particular local directory. Both CPA and CELMAX are from
the original FedDistill project; this change does not introduce their algorithms.

## Fixes

- Explicit `public_ds=stl10` now selects `stl10-cov.pt`, rather than the private
  dataset's file name. In-domain public subsets use the private dataset's name.
  A missing in-domain matrix now raises an actionable error rather than silently
  substituting the STL-10 matrix. `--cpa_cov_path` permits a supplied matrix.
- Bundled matrices are resolved relative to `attacks.py`, independently of cwd.
  The matrix must be finite, floating point, square, symmetric, and match the
  classifier's class count. Shape alone does not establish class ordering: custom
  matrices must use the private classifier's class order, not the public dataset's
  native ground-truth label order. No test labels are used to select attack targets.
- CPA now has its own raw-logit entry point. Honest logits and warmup uploads are
  returned unchanged. The previous generic path converted every client's
  probabilities back to log-probabilities. At T=1 and without clipping, that is a
  per-sample scalar shift of honest logits, which cancels under Mean + softmax;
  distance-based robust aggregators need not be invariant to these shifts.
- Partial attacks now preserve each client's original uploads on untouched
  samples, including an exact no-op for `sample_attack_frac=0`. This intentionally
  replaces the old honest-mean backfill. The attacked count is floor(N*fraction).
  Random masks use the isolated attack RNG; they do not consume training RNG.
- Invalid fractions, epsilon, non-FedDistill strategies, and missing honest
  reference clients fail explicitly. HIPS still requires full-sample attacks,
  as in the original implementation; its sample indices now use the tensor device.
- Ordinary CPA still selects the class minimizing the covariance row indexed by
  the honest consensus class. On the bundled symmetric covariance, this agrees
  with the original column-based argmin. HIPS still selects an honest prediction
  vertex per public sample to minimize the same covariance objective. On the
  raw-logit path, HIPS uploads the selected honest client's actual logits,
  preserving the honest-vertex constraint in the space being aggregated.

## Protocol limits

This implementation is an **omniscient, logit-space adaptation**. It observes
honest clients' public predictions; this information is not normally available to
a real malicious client. Do not describe it as a restricted-knowledge attack.

For ordinary CPA, the malicious probabilities are mapped to logits using
`log(max(p, cpa_logit_epsilon))`. The default remains 1e-12, so a one-hot row
becomes approximately 0 for the target and -27.631 for all other classes.
This is an explicitly recorded adaptation parameter, not a parameter claimed by
the original paper. Probability averaging and logit averaging differ, so these
runs do not reproduce the original probability-aggregation results exactly.
HIPS selects and uploads an actual honest raw-logit vector for each sample;
epsilon does not control its amplitude.

The script fixes temperature to 1. Other temperatures change the effective
attack distribution. Whether CPA actually lowers accuracy or evades SPC must
be measured; successful execution does not establish either claim.

## Experiment plan

Run from the project root on the Linux server:

```bash
bash ssh/run_cpa_spc_matrix.sh 12101 full plan plain
```

| Run ID | Attack | Aggregation |
| --- | --- | --- |
| 12101 | None | Mean |
| 12102 | None | SPC then Mean |
| 12103 | CPA, ordinary variant | Mean |
| 12104 | CPA, ordinary variant | SPC then Mean |

Each group starts independently from random initialization with seed 20260805.
This does not resume 11301 or any other historical checkpoint. Full settings:

- CIFAR-10 private train 47500, validation 2500, test 10000;
  STL-10 unlabeled public 100000; ResNet18.
- 30 clients, 9 malicious in attack groups and 0 in clean groups.
- 70 total rounds; normal training rounds 1-14; attack and probe audit/filtering
  start at round 15. Local training and server KD each use one epoch per round.
- SGD, momentum .9, weight decay .0001, batch 128; server/client Linear .1 LR;
  no pretraining, client public KD, purifier, or AMP; temperature 1.
- Identical scale/replace probes, 20 groups x 4 intensities (1, 1.2, 1.5, 2),
  probe seed 0; SPC core threshold 5. Probe inputs/audits are enabled in all runs;
  only filtering is toggled. Attack sample fraction 1, epsilon 1e-12.

The same seed does not substitute for multiple seeds. Attack success is the
drop relative to the clean group with the same defense. Evaluate server clean
accuracy, final/last-round stability, malicious upload retention and honest false
positives; do not presume filtering will work.

First run the two-round execution check (still using the full public dataset):

```bash
bash ssh/run_cpa_spc_matrix.sh 12091 smoke run plain
```

Then start the full experiment detached:

```bash
mkdir -p logs
nohup bash ssh/run_cpa_spc_matrix.sh 12101 full run plain > logs/cpa_12101.launcher.log 2>&1 < /dev/null &
tail -f logs/cpa_12101.launcher.log
```

Detailed progress is in `logs/12101_clear_mean.log`, etc. The launcher prints
which file is active. The final summary is
`logs/12101_12104_cpa_plain_full.summary.txt`. Existing IDs are never overwritten.
The script verifies the CPA tests, matrix and existing dataset integrity before
training, and stops if another scheduler/training process is active.

To use a different server interpreter:

```bash
PYTHON=/path/to/python bash ssh/run_cpa_spc_matrix.sh 12101 full run plain
```

HIPS is a separate optional experiment, not started by the commands above:

```bash
bash ssh/run_cpa_spc_matrix.sh 12201 full plan hips
# Replace plan with run only when you intend to run this second four-group matrix.
```

## Upload from Windows PowerShell

The helper uploads only the listed code, CPA covariance, test, script and this
document, and checks remote SHA-256 hashes. It assumes the full project and its
Python dependencies already exist on the server. It does not start training.

```powershell
& .\ssh\upload_cpa.ps1 -Server 'user@your-server' -RemoteDir '/path/to/fdd-main'
# Optional: -Port 2222 -IdentityFile 'C:\path\to\key'
```

## Local validation

```powershell
python -m unittest discover -s tests -p test_cpa.py -v
```

The local Windows environment has CPU PyTorch. Regression tests and launch plan
checks do not replace the server execution check or a completed 70-round run.
