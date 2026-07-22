# Super Poker 3

An open-source Poker44 SN126 miner using a regularized XGBoost bot detector.

## Design

- miner-visible, sanitization-aware behavior features only;
- hero-relative features plus robust all-table context;
- entropy, variability, quantiles, response behavior, and cross-hand signatures;
- chronological walk-forward evaluation on unseen release dates;
- deployment threshold learned from prior-date human out-of-fold scores;
- independent per-chunk probabilities with no top-k or prevalence forcing;
- current `poker44.score.scoring.reward` used for evaluation.

No performance on private live data is guaranteed. The saved metrics describe public
benchmark validation only.

## Baseline Result

The artifact trained on 2026-07-13 used five chronological walk-forward folds
(2026-07-09 through 2026-07-13). Each fold trained only on earlier dates and learned
its threshold from an earlier calibration release.

| Metric | Result |
|---|---:|
| Poker44 reward | 0.8749 |
| Average precision | 0.9341 |
| ROC AUC | 0.9298 |
| Bot recall at constrained FPR | 0.6600 |
| Observed FPR | 0.0486 |
| Hard bot recall at 0.5 | 0.6429 |
| Hard human FPR at 0.5 | 0.0457 |

These results are model-selection evidence on public releases. They are not evidence
that the final all-data artifact has seen or will win on private validator batches.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

The trainer reads the public cache created by `Poker44-subnet/bot_detector/download.py`.
The default location is `../Poker44-subnet/data/raw`.

## Train

```bash
python -m super_poker.train \
  --data-dir ../Poker44-subnet/data/raw \
  --artifact artifacts/super_poker_3.joblib
```

Training writes:

- `artifacts/super_poker_3.joblib`: model, feature schema, threshold, metadata;
- `artifacts/super_poker_3.metrics.json`: readable validation and provenance data.

## Evaluate

```bash
python -m super_poker.evaluate --dates 2026-07-12,2026-07-13
```

Do not treat evaluation on releases used for final training as an unseen test result. The
walk-forward metrics embedded during training are the honest model-selection signal.

## Run Miner

After training, use the standard Poker44 miner command or scripts. Override the artifact with:

```bash
export SUPER_POKER_MODEL_PATH=/absolute/path/to/super_poker_3.joblib
```

Before publishing, set the real public repository URL in `neurons/miner.py` so the model
manifest can meet transparent-miner policy.

## Automatic Learning

Automation uses `../Poker44-subnet/data/raw` by default, matching the training command.
That cache should contain the historical backfill created during initial setup. To initialize
an empty cache directly through this project, run once with an explicit directory:

```bash
python -m super_poker.automation daily --data-dir data/raw --backfill
```

Daily data-only update:

```bash
python -m super_poker.automation daily
```

Daily update plus a non-deployed candidate:

```bash
python -m super_poker.automation daily --train-candidate
```

The proven single-XGBoost family remains the automated default. To run the
experimental multi-seed XGBoost + ExtraTrees challenger without deployment:

```bash
SUPER_POKER_MODEL_FAMILY=ensemble \
python -m super_poker.automation daily --train-candidate --no-download
```

Every request logs aggregate score ranges, threshold-crossing rate, chunk-size
range, and latency. Raw hand payloads are not persisted by this diagnostic path.

Competition-cycle retraining and guarded deployment:

```bash
python -m super_poker.automation cycle
```

The cycle checker is anchored at `2026-07-16 12:00 UTC`, repeats every 120 hours, and
starts the guarded workflow six hours before the upcoming competition. It can be configured:

```bash
export SUPER_POKER_CYCLE_ANCHOR_UTC=2026-07-16T12:00:00Z
export SUPER_POKER_CYCLE_HOURS=120
export SUPER_POKER_CYCLE_LEAD_HOURS=6
```

The cycle candidate is approved and promoted only when all excellent-performance checks pass:

- reward >= 0.85 and no more than 0.002 below the incumbent;
- average precision >= 0.92 and no more than 0.002 below the incumbent;
- pooled FPR <= 0.05;
- hard-threshold FPR <= 0.06;
- every walk-forward fold reward >= 0.80.

Rejected candidates remain under `artifacts/candidates/`. Approved candidates are copied to
`artifacts/approved/` with metrics and the gate decision. Successful deployment then copies
the incumbent to `artifacts/backups/` and atomically replaces the serving artifact.
The latest decision is recorded in `artifacts/automation-state.json`.

### VPS daily candidate training

The preferred production schedule downloads the latest public release, trains a candidate,
evaluates it, and records the decision without changing the serving artifact or restarting PM2.
Choose a UTC time after the daily evaluation normally finishes, then install the supplied units:

```bash
sudo cp scripts/model/super-poker-3-daily-candidate.service /etc/systemd/system/
sudo cp scripts/model/super-poker-3-daily-candidate.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now super-poker-3-daily-candidate.timer
```

The checked-in service expects the VPS checkout at `/root/super_poker_3` and its Python at
`/root/super_poker_3/.venv/bin/python`. Adjust those paths before installation when necessary.
The timer defaults to `03:15 UTC`, uses a five-minute randomized delay, and limits training to 40%
CPU with idle I/O priority. Confirm that this time does not overlap the subnet's daily evaluation.

```bash
systemctl list-timers super-poker-3-daily-candidate.timer
journalctl -u super-poker-3-daily-candidate.service -n 100 --no-pager
jq '{deployed, decision, data}' artifacts/automation-state.json
```

`vps_daily_candidate.sh` uses `flock` to prevent overlapping runs, archives each decision under
`artifacts/daily-history/`, and fails if daily mode changes the incumbent artifact hash. It never
restarts PM2. An approved candidate still requires a manual source/artifact/manifest review and
an explicit deployment decision between locked Poker44 v2.0 rounds.

Do not schedule `weekly_learning.sh` or `cycle_learning.sh` unattended on a competition miner;
those modes can promote an approved artifact. `scripts/model/crontab` contains a legacy,
non-deploying cron alternative for hosts that do not use systemd.

## License

MIT. Poker44 reference code remains under its original MIT license.
