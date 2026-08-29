"""Silicon baseline: can any learner extract edge from what the neurons see?

If a conventional model with far more learning capacity than a neural culture
cannot find signal in the encoded features at game cadence, the game design
needs fixing before spending wetware rental time on it. This package builds
datasets from recordings or synthetic markets, trains cheap probes (logistic,
GPU MLP), and backtests the resulting policies through the real game engine
and paper broker so costs are honest.
"""

from .dataset import BaselineDataset, build_dataset, load_recording_series, synthetic_series
from .models import binomial_margin, fit_logistic, fit_mlp

__all__ = [
    "BaselineDataset",
    "build_dataset",
    "load_recording_series",
    "synthetic_series",
    "fit_logistic",
    "fit_mlp",
    "binomial_margin",
]
