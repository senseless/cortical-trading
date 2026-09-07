"""Session metrics: per-episode and session-level scores from step logs."""

from __future__ import annotations

from collections import Counter


def _directional_accuracy(steps: list[dict]) -> float | None:
    """How often did an acting decision match the move over the NEXT step?

    Deliberately narrow, and not the objective: it scores one step (~1 s) of
    foresight, while the game is won or lost on multi-minute holds after
    costs. A culture can be right here and still lose money (the
    accurate-but-untradeable trap), which is why pnl_after_costs is the
    default score_metric. It survives as a cheap, cost-free signal check.

    Only decision rows count, and only pairs that are actually one step apart:
    flatten/roll rows are forced, not predictions, and consecutive log rows can
    straddle a feedback pause or a rest, where the "next" mid is tens of
    seconds away and measures nothing the decision could have known.
    """
    rows = [r for r in steps if r.get("step", 0) >= 0]
    if len(rows) < 3:
        return None
    gaps = sorted(b["t"] - a["t"] for a, b in zip(rows[:-1], rows[1:]))
    step_s = gaps[len(gaps) // 2]  # median spacing = the session's step cadence
    limit = 1.5 * step_s
    hits = total = 0
    for row, nxt_row in zip(rows[:-1], rows[1:]):
        side = {"buy": 1, "sell": -1}.get(row["action"], 0)
        if side == 0 or nxt_row["t"] - row["t"] > limit:
            continue
        move = nxt_row["mid"] - row["mid"]
        if move == 0:
            continue
        total += 1
        hits += int((move > 0) == (side > 0))
    return hits / total if total else None


def episode_metrics(steps: list[dict]) -> dict:
    # Flatten/roll rows are logged with step=-1: they carry real trades and
    # feedback but are not decisions the culture made, so they are excluded
    # from the step count and the action histogram.
    decisions = [row for row in steps if row.get("step", 0) >= 0]
    actions = Counter(row["action"] for row in decisions)
    feedback = Counter(row["feedback_kind"] for row in steps if row.get("feedback_kind"))
    # Trades from every row, including forced closes: a flatten's PnL is real
    # money and belongs in the episode's result.
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
