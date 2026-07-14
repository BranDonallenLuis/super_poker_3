import math

from super_poker.features import chunk_features, hand_features


def test_empty_features_are_stable_and_finite():
    assert set(chunk_features([])) == set(chunk_features([{}]))
    assert all(math.isfinite(value) for value in chunk_features([]).values())


def test_hero_behavior_is_separate_from_opponents():
    hand = {
        "metadata": {"hero_seat": 2, "max_seats": 6},
        "players": [{"seat": 1, "starting_stack": 2}, {"seat": 2, "starting_stack": 4}],
        "actions": [
            {"street": "preflop", "actor_seat": 1, "action_type": "raise", "amount": 1, "pot_before": 1},
            {"street": "preflop", "actor_seat": 2, "action_type": "raise", "amount": 3, "pot_before": 2},
            {"street": "flop", "actor_seat": 1, "action_type": "bet", "amount": 1, "pot_before": 4},
            {"street": "flop", "actor_seat": 2, "action_type": "fold", "amount": 0, "pot_before": 5},
        ],
    }
    values = hand_features(hand)
    assert values["three_bet"] == 1.0
    assert values["pfr"] == 1.0
    assert values["fold_to_aggression"] == 0.5
    assert values["hero_raise_count"] == 1.0
    assert values["hero_stack_vs_table"] > 1.0


def test_labels_and_ids_do_not_change_features():
    base = {"metadata": {"hero_seat": 1}, "actions": []}
    leaked = {**base, "hand_id": "special", "label": "bot", "is_bot": True}
    assert hand_features(base) == hand_features(leaked)
