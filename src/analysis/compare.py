"""Statistical comparison of two sessions (e.g. experiment vs. random control).

Per-episode scores are the unit of observation, compared with Mann-Whitney U
(no normality assumption; matches the small-n, non-normal reality of culture
experiments). Never conclude from a single session -- collect several.
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
        result["note"] = ("difference unlikely to be chance" if p < 0.05
                          else "no significant difference (collect more sessions before concluding)")
    else:
        result["note"] = "need >= 2 episodes per session for a test"
    return result
