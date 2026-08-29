"""Statistical comparison of two sessions (e.g. experiment vs. random control).

Per-episode scores are compared with Mann-Whitney U (no normality assumption),
but episodes within one session share a culture, a day, and one market path:
they are NOT independent samples, so a p-value from a single session pair is
pseudoreplicated and descriptive only. The unit of replication for any claim
about learning is the session (ideally the culture), never the episode.
"""

from __future__ import annotations

from pathlib import Path

from scipy import stats

from .session_io import load_session


def _episode_scores(sess: dict, metric: str) -> list[float]:
    episodes = sess["metrics"].get("episodes", {})
    values = []
    for ep in sorted(episodes, key=int):
        v = episodes[ep].get(metric)
        if v is not None:
            values.append(float(v))
    return values


def compare_sessions(dir_a: str | Path, dir_b: str | Path,
                     metric: str = "pnl_dollars") -> dict:
    a, b = load_session(dir_a), load_session(dir_b)
    sa, sb = _episode_scores(a, metric), _episode_scores(b, metric)
    result = {
        "metric": metric,
        "a": {"dir": str(a["dir"]), "label": a["metrics"].get("label"), "episodes": sa,
              "mean": sum(sa) / len(sa) if sa else None},
        "b": {"dir": str(b["dir"]), "label": b["metrics"].get("label"), "episodes": sb,
              "mean": sum(sb) / len(sb) if sb else None},
    }
    if len(sa) >= 2 and len(sb) >= 2:
        u, p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        result["mannwhitney_u"] = float(u)
        result["p_value"] = float(p)
        # Episodes within a session are not independent (one culture, one
        # market path), so this p-value cannot support a significance claim
        # on its own -- treating it as confirmatory would be pseudoreplication.
        result["caveat"] = ("episodes are pseudoreplicates: p-value is descriptive of this "
                            "session pair only; conclude learning from replication across "
                            "sessions/cultures, not from this test")
        result["note"] = (f"episode scores differ within this pair (p={p:.3f}); "
                          "replicate across sessions before drawing conclusions" if p < 0.05
                          else "no detectable difference in this pair (and single pairs "
                               "cannot confirm one anyway -- collect more sessions)")
    else:
        result["note"] = "need >= 2 episodes per session for a test"
    return result
