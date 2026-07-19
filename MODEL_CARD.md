# Super Poker 3 Model Card

## Live-score-driven development

Competition scores are recorded against the exact model version and artifact hash in
`config/live_scores.json`. They guide distribution-shift investigation and feature design,
but are never treated as per-example labels. The R3 reform adds exact validator-visible
amount buckets, hero-independent action/actor/street/amount signatures, and same-date,
same-label 90-105-hand training augmentation while retaining real-only chronological tests.

## Model identity

- Model: `super-poker-3-xgboost`
- Version: `20260713-185316`
- Framework: XGBoost
- Artifact: `super_poker_3.joblib`
- Artifact URL: <https://github.com/BranDonallenLuis/super_poker_3/releases/download/model-20260713-185316/super_poker_3.joblib>
- Artifact SHA-256: `f607abdeea01419631a6d5e03b870eb95c4572b3ecd1e198c8feb6890b159f53`
- Feature schema: `super-poker-3.v1` (435 features)
- Feature-schema SHA-256: `bbd55dbffe6b31d0574389e8f7b203c84f3346f406cda7400dc5ef8dd9f8bfba`

## Training data statement

Trained on 1,740 examples from Poker44 public benchmark releases dated 2026-05-26
through 2026-07-13. No validator-private data or labels were used. The final five
release dates, 2026-07-09 through 2026-07-13, were evaluated chronologically using
walk-forward validation.

## Validation metrics

The overall walk-forward results recorded in `artifacts/super_poker_3.metrics.json` were:

| Metric | Value |
| --- | ---: |
| Reward | 0.874946 |
| Average precision | 0.934133 |
| ROC AUC | 0.929796 |
| False-positive rate | 0.048571 |
| Bot recall | 0.660000 |
| Hard false-positive rate | 0.045714 |
| Hard bot recall | 0.642857 |
| Brier score | 0.124732 |
| Log loss | 0.390087 |

These metrics are measurements on public benchmark releases, not a guarantee of live
competition performance. Live data distribution and validator behavior may differ.

## Artifact verification

After downloading the release asset, verify it before use:

```bash
echo "f607abdeea01419631a6d5e03b870eb95c4572b3ecd1e198c8feb6890b159f53  super_poker_3.joblib" | sha256sum --check
```
