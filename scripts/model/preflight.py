#!/usr/bin/env python3
"""Refuse deployment when the model artifact and runtime features disagree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_poker.features import chunk_features
from super_poker.inference import SuperPokerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/super_poker_3.joblib"))
    parser.add_argument("--expected-model", default="super-poker-3-xgboost-enhanced")
    args = parser.parse_args()

    model = SuperPokerModel(args.artifact)
    runtime = set(chunk_features([]))
    required = set(model.feature_names)
    missing = sorted(required - runtime)
    extra = sorted(runtime - required)
    scores = model.predict_chunk_scores([[], [{}]])
    sha256 = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    ok = (
        not missing
        and not extra
        and model.metadata.get("model_name") == args.expected_model
        and len(scores) == 2
        and all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores)
    )
    print(json.dumps({
        "ok": ok,
        "model_name": model.metadata.get("model_name"),
        "model_version": model.metadata.get("model_version"),
        "artifact_sha256": sha256,
        "artifact_features": len(required),
        "runtime_features": len(runtime),
        "missing_features": missing,
        "extra_features": extra,
        "smoke_scores": scores,
    }, indent=2))
    if not ok:
        raise SystemExit("Super Poker 3 preflight failed; refusing deployment")


if __name__ == "__main__":
    main()
