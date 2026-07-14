"""Sanitization-aware, size-stable behavioral features for Poker44 chunks."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

ACTIONS = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def _mean(values: list[float]) -> float:
    return _divide(sum(values), len(values))


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(max(0.0, _mean([(value - mean) ** 2 for value in values])))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _entropy(values: Iterable[Any]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    raw = -sum((count / total) * math.log(count / total) for count in counts.values())
    return _divide(raw, math.log(len(counts)))


def _switch_rate(values: list[Any]) -> float:
    return _divide(sum(a != b for a, b in zip(values, values[1:])), len(values) - 1)


def _top_share(values: list[Any]) -> float:
    return _divide(max(Counter(values).values()), len(values)) if values else 0.0


def _amount_bucket(value: float) -> int:
    limits = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 36.0, 84.0, 126.0)
    return next((index for index, limit in enumerate(limits) if value <= limit), len(limits))


def hand_features(hand: dict[str, Any]) -> dict[str, float]:
    """Extract numeric features without using IDs, labels, dates, or hashes."""
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    players = hand.get("players") if isinstance(hand.get("players"), list) else []
    streets = hand.get("streets") if isinstance(hand.get("streets"), list) else []
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    outcome = hand.get("outcome") if isinstance(hand.get("outcome"), dict) else {}
    hero = int(_number(metadata.get("hero_seat")))
    max_seats = max(1, int(_number(metadata.get("max_seats")) or 6))

    types = [str(a.get("action_type") or "").lower() for a in actions]
    actors = [int(_number(a.get("actor_seat"))) for a in actions]
    action_streets = [str(a.get("street") or "").lower() for a in actions]
    hero_actions = [a for a in actions if int(_number(a.get("actor_seat"))) == hero and hero > 0]
    hero_types = [str(a.get("action_type") or "").lower() for a in hero_actions]
    hero_streets = [str(a.get("street") or "").lower() for a in hero_actions]
    counts, hero_counts = Counter(types), Counter(hero_types)
    meaningful = sum(counts[action] for action in ACTIONS)
    hero_meaningful = sum(hero_counts[action] for action in ACTIONS)

    amounts = [max(0.0, _number(a.get("normalized_amount_bb"))) for a in hero_actions]
    aggressive_amounts = [
        max(0.0, _number(a.get("normalized_amount_bb")))
        for a in hero_actions
        if str(a.get("action_type") or "").lower() in {"bet", "raise"}
    ]
    pot_ratios = []
    for action in hero_actions:
        amount = max(0.0, _number(action.get("amount")))
        pot = max(0.0, _number(action.get("pot_before")))
        if amount > 0 and pot > 0:
            pot_ratios.append(min(20.0, amount / pot))

    stacks = [max(0.0, _number(p.get("starting_stack"))) for p in players if isinstance(p, dict)]
    hero_stack = next(
        (max(0.0, _number(p.get("starting_stack"))) for p in players
         if isinstance(p, dict) and int(_number(p.get("seat"))) == hero),
        0.0,
    )
    preflop_hero = [a for a in hero_actions if str(a.get("street") or "").lower() == "preflop"]
    preflop_types = [str(a.get("action_type") or "").lower() for a in preflop_hero]
    prior_preflop_raises = 0
    three_bet = 0.0
    for action in actions:
        if str(action.get("street") or "").lower() != "preflop":
            continue
        kind = str(action.get("action_type") or "").lower()
        actor = int(_number(action.get("actor_seat")))
        if kind == "raise" and actor == hero and prior_preflop_raises:
            three_bet = 1.0
        if kind == "raise":
            prior_preflop_raises += 1

    faced, response_fold, response_call, response_raise = 0, 0, 0, 0
    pending = False
    for action in actions:
        actor = int(_number(action.get("actor_seat")))
        kind = str(action.get("action_type") or "").lower()
        if actor != hero and kind in {"bet", "raise"}:
            pending = True
        elif actor == hero and pending:
            faced += 1
            response_fold += kind == "fold"
            response_call += kind == "call"
            response_raise += kind == "raise"
            pending = False

    result: dict[str, float] = {
        "player_count": float(len(players)),
        "seat_utilization": _divide(len(players), max_seats),
        "street_count": float(len(streets)),
        "table_action_count": float(len(actions)),
        "hero_action_count": float(len(hero_actions)),
        "hero_action_share": _divide(len(hero_actions), len(actions)),
        "action_entropy": _entropy(types),
        "actor_entropy": _entropy(actors),
        "street_entropy": _entropy(action_streets),
        "action_switch_rate": _switch_rate(types),
        "actor_switch_rate": _switch_rate(actors),
        "hero_action_entropy": _entropy(hero_types),
        "hero_action_switch_rate": _switch_rate(hero_types),
        "aggression_rate": _divide(hero_counts["bet"] + hero_counts["raise"], hero_meaningful),
        "aggression_factor": _divide(hero_counts["bet"] + hero_counts["raise"], hero_counts["call"]),
        "vpip": float(any(kind in {"call", "bet", "raise"} for kind in preflop_types)),
        "pfr": float("raise" in preflop_types),
        "three_bet": three_bet,
        "faced_aggression": float(faced),
        "fold_to_aggression": _divide(response_fold, faced),
        "call_vs_aggression": _divide(response_call, faced),
        "raise_vs_aggression": _divide(response_raise, faced),
        "amount_mean_bb": _mean(amounts),
        "amount_std_bb": _std(amounts),
        "aggressive_amount_mean_bb": _mean(aggressive_amounts),
        "aggressive_amount_std_bb": _std(aggressive_amounts),
        "pot_ratio_mean": _mean(pot_ratios),
        "pot_ratio_std": _std(pot_ratios),
        "hero_stack": hero_stack,
        "stack_mean": _mean(stacks),
        "stack_std": _std(stacks),
        "hero_stack_vs_table": _divide(hero_stack, _mean(stacks)),
        "showdown": float(bool(outcome.get("showdown"))),
        "zero_hero_actions": float(not hero_actions),
    }
    for action in ACTIONS:
        result[f"table_{action}_share"] = _divide(counts[action], meaningful)
        result[f"hero_{action}_count"] = float(hero_counts[action])
        result[f"hero_{action}_share"] = _divide(hero_counts[action], hero_meaningful)
    for street in STREETS:
        result[f"table_{street}_share"] = _divide(action_streets.count(street), len(actions))
        result[f"hero_{street}_action_count"] = float(hero_streets.count(street))
        result[f"hero_reached_{street}"] = float(street in hero_streets)
    return result


def _aggregate(name: str, values: list[float], output: dict[str, float]) -> None:
    output[f"{name}_mean"] = _mean(values)
    output[f"{name}_std"] = _std(values)
    output[f"{name}_min"] = min(values) if values else 0.0
    output[f"{name}_max"] = max(values) if values else 0.0
    output[f"{name}_q10"] = _quantile(values, 0.10)
    output[f"{name}_q50"] = _quantile(values, 0.50)
    output[f"{name}_q90"] = _quantile(values, 0.90)


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate hands and cross-hand regularity into one fixed feature row."""
    hands = [hand for hand in (chunk or []) if isinstance(hand, dict)]
    rows = [hand_features(hand) for hand in hands]
    template = hand_features({})
    output: dict[str, float] = {"hand_count": float(len(hands)), "log_hand_count": math.log1p(len(hands))}
    for name in sorted(template):
        _aggregate(name, [row[name] for row in rows], output)

    action_signatures, role_signatures, amount_signatures = [], [], []
    for hand in hands:
        metadata = hand.get("metadata") or {}
        hero = int(_number(metadata.get("hero_seat")))
        actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
        action_signatures.append(tuple(str(a.get("action_type") or "").lower() for a in actions))
        role_signatures.append(tuple("H" if int(_number(a.get("actor_seat"))) == hero else "O" for a in actions))
        amount_signatures.append(tuple(_amount_bucket(max(0.0, _number(a.get("normalized_amount_bb")))) for a in actions))
    for name, signatures in (
        ("action_signature", action_signatures),
        ("role_signature", role_signatures),
        ("amount_signature", amount_signatures),
    ):
        output[f"{name}_top_share"] = _top_share(signatures)
        output[f"{name}_unique_share"] = _divide(len(set(signatures)), len(signatures))
    return {name: value if math.isfinite(value) else 0.0 for name, value in output.items()}
