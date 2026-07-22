import joblib
import numpy as np

from super_poker.ensemble import ProbabilityEnsemble
from super_poker.feature_policy import validator_stable_features
from super_poker.inference import SuperPokerModel
from super_poker.train import make_model


class FixedModel:
    def predict_proba(self, frame):
        return np.asarray([[0.2, 0.8] for _ in range(len(frame))])


def test_inference_returns_one_bounded_score_per_chunk(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": FixedModel(), "feature_names": ["hand_count"], "threshold": 0.7}, path)
    scores = SuperPokerModel(path).predict_chunk_scores([[], [{}]])
    assert len(scores) == 2
    assert all(0.5 < score <= 1.0 for score in scores)


def test_inference_exposes_privacy_safe_score_components(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": FixedModel(), "feature_names": ["hand_count"], "threshold": 0.7}, path)
    components = SuperPokerModel(path).predict_chunk_components([[]])
    assert components == {"raw_scores": [0.8], "final_scores": [components["final_scores"][0]]}
    assert 0.5 < components["final_scores"][0] <= 1.0


def test_probability_ensemble_normalizes_weights():
    ensemble = ProbabilityEnsemble([FixedModel(), FixedModel()], [3.0, 1.0])
    probabilities = ensemble.predict_proba(np.zeros((2, 1)))
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities[:, 1], 0.8)


def test_feature_policy_drops_known_drift_fields():
    names = ["hand_count", "hero_raise_share_mean", "actor_entropy_mean", "table_fold_share_q50"]
    assert validator_stable_features(names) == ["actor_entropy_mean", "table_fold_share_q50"]


def test_xgboost_profiles_are_explicit_and_validated():
    assert make_model(profile="robust").get_params()["n_estimators"] == 320
    try:
        make_model(profile="unknown")
    except ValueError as exc:
        assert "Unknown XGBoost profile" in str(exc)
    else:
        raise AssertionError("unknown profile should fail")
