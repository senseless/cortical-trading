"""Per-session analysis: learning curves, PnL, action-reward alignment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .session_io import load_session  # noqa: E402


def action_reward_alignment(steps: list[dict]) -> dict:
    """How feedback related to behavior; sanity checks for the learning signal."""
    acting = [s for s in steps if s["action"] in ("buy", "sell")]
    feedback = [s for s in steps if s.get("feedback_kind")]
    consistent = sum(
        1 for s in feedback
        if (s["feedback_kind"] == "predictable") == (s["reward"] > 0)
    )
    return {
        "n_steps": len(steps),
        "n_acting": len(acting),
        "acting_fraction": round(len(acting) / len(steps), 4) if steps else None,
        "n_feedback": len(feedback),
        "n_predictable": sum(1 for s in feedback if s["feedback_kind"] == "predictable"),
        "n_unpredictable": sum(1 for s in feedback if s["feedback_kind"] == "unpredictable"),
        # 1.0 unless reward.shuffle was on (the control decouples feedback from outcomes)
        "feedback_outcome_consistency": round(consistent / len(feedback), 4) if feedback else None,
    }


def analyze_session(session_dir: str | Path) -> dict:
    sess = load_session(session_dir)
    steps, metrics = sess["steps"], sess["metrics"]
    out = Path(session_dir) / "analysis"
    out.mkdir(exist_ok=True)

    episodes = metrics.get("episodes", {})
    ep_ids = sorted(episodes, key=int)
    pnl = [episodes[e]["pnl_dollars"] for e in ep_ids]
    acc = [episodes[e]["directional_accuracy"] for e in ep_ids]

    # Learning curve: per-episode PnL and directional accuracy.
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(range(len(ep_ids)), pnl, color=["#2a9d8f" if p >= 0 else "#e76f51" for p in pnl])
    ax1.set_xlabel("episode")
    ax1.set_ylabel("PnL after costs ($)")
    ax1.axhline(0, color="gray", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(range(len(ep_ids)), [a if a is not None else float("nan") for a in acc],
             "o-", color="#264653", label="directional accuracy")
    ax2.axhline(0.5, color="#264653", lw=0.8, ls="--", alpha=0.5)
    ax2.set_ylabel("directional accuracy")
    ax2.set_ylim(0, 1)
    ax1.set_title(f"learning curve: {metrics.get('label', '?')} ({metrics.get('market', '?')})")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=120)
    plt.close(fig)

    # Cumulative realized PnL across the session, with episode boundaries.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = range(len(steps))
    ax.plot(xs, [s["realized_dollars"] for s in steps], color="#2a9d8f")
    boundaries = [i for i in xs if i > 0 and steps[i]["episode"] != steps[i - 1]["episode"]]
    for b in boundaries:
        ax.axvline(b, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("realized PnL ($)")
    ax.set_title("cumulative realized PnL")
    fig.tight_layout()
    fig.savefig(out / "cumulative_pnl.png", dpi=120)
    plt.close(fig)

    alignment = action_reward_alignment(steps)
    summary = {
        "session": str(sess["dir"]),
        "label": metrics.get("label"),
        "mode": metrics.get("mode"),
        "market": metrics.get("market"),
        "score": metrics.get("score"),
        "pnl_dollars": metrics.get("pnl_dollars"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "n_trades": metrics.get("n_trades"),
        "alignment": alignment,
        "plots": [str(out / "learning_curve.png"), str(out / "cumulative_pnl.png")],
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
