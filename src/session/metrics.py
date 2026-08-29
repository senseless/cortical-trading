"""Session metrics: per-episode and session-level scores from step logs."""

from __future__ import annotations

from collections import Counter


def _directional_accuracy(steps: list[dict]) -> float | None:
    """Among acting steps, how often did the action match the next price move?"""
    hits = total = 0
    for i, row in enumerate(steps[:-1]):
        side = {"buy": 1, "sell": -1}.get(row["action"], 0)
        if side == 0:
            continue
        nxt = steps[i + 1]["mid"] - row["mid"]
        if nxt == 0:
            continue
        total += 1
        hits += int((nxt > 0) == (side > 0))
    return hits / total if total else None


def episode_metrics(steps: list[dict]) -> dict:
    # Flatten/roll rows are logged with step=-1: they carry real trades and
    # feedback but are not decisions the culture made, so they are excluded
    # from the step count and the action histogram.
    decisions = [row for row in steps if row.get("step", 0) >= 0]
    actions = Counter(row["action"] for row in decisions)
    feedback = Counter(row["feedback_kind"] for row in steps if row.get("feedback_kind"))
    closed = [row["closed_trade"] for row in steps if row.get("closed_trade")]
    wins = [t for t in closed if t["dollars"] > 0]
    pnl_dollars = sum(t["dollars"] for t in closed)
    pnl_points = sum(t["points"] for t in closed)
    return {
        "steps": len(decisions),
        "pnl_points": round(pnl_points, 4),
        "pnl_dollars": round(pnl_dollars, 2),
        "n_trades": len(closed),
        "win_rate": round(len(wins) / len(closed), 4) if closed else None,
        "directional_accuracy": _directional_accuracy(steps),
        "actions": dict(actions),
        "feedback": dict(feedback),
    }


def session_metrics(steps: list[dict], score_metric: str) -> dict:
    episodes: dict[int, list[dict]] = {}
    for row in steps:
        episodes.setdefault(row["episode"], []).append(row)
    per_episode = {ep: episode_metrics(rows) for ep, rows in sorted(episodes.items())}

    total_dollars = sum(m["pnl_dollars"] for m in per_episode.values())
    total_points = sum(m["pnl_points"] for m in per_episode.values())
    total_trades = sum(m["n_trades"] for m in per_episode.values())
    accs = [m["directional_accuracy"] for m in per_episode.values() if m["directional_accuracy"] is not None]
    directional = sum(accs) / len(accs) if accs else None

    if score_metric == "directional_accuracy":
        score = directional if directional is not None else 0.0
    else:
        score = total_dollars

    return {
        "score_metric": score_metric,
        "score": score,
        "pnl_points": round(total_points, 4),
        "pnl_dollars": round(total_dollars, 2),
        "n_trades": total_trades,
        "directional_accuracy": directional,
        "episodes": per_episode,
    }
