"""CL API access. Sets simulator environment variables before importing cl.

The CL API is identical on the simulator (cl-sdk) and on a real CL1, so this
is the only place aware of the difference.
"""

from __future__ import annotations

import logging
import os
import sys

from ..config import NeuralCfg

log = logging.getLogger(__name__)


def get_cl(cfg: NeuralCfg, accelerated: bool = False):
    if "cl" in sys.modules:
        # The env vars below only apply at import time; a pre-imported module
        # keeps whatever settings it started with.
        log.warning("cl was already imported; sdk_seed/accelerated settings may not take effect")
    os.environ.setdefault("CL_SDK_VISUALISATION", "0")
    if cfg.sdk_seed >= 0:
        os.environ["CL_SDK_RANDOM_SEED"] = str(cfg.sdk_seed)
    else:
        # -1 means unseeded: clear any seed inherited from a previous run in
        # this process (or the shell), otherwise "unseeded" silently repeats
        # the same simulator randomness.
        os.environ.pop("CL_SDK_RANDOM_SEED", None)
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1" if accelerated else "0"
    import cl

    return cl


def backend_name(cl_mod) -> str:
    try:
        sim = cl_mod.is_simulator() if callable(cl_mod.is_simulator) else bool(cl_mod.is_simulator)
    except Exception:
        # Never guess: provenance ("was this real neurons?") must not default
        # to a wrong answer if the probe fails on some SDK version.
        return "unknown (is_simulator probe failed)"
    return "cl-sdk simulator" if sim else "CL1 hardware"
