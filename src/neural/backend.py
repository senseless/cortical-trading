"""CL API access. Sets simulator environment variables before importing cl.

The CL API is identical on the simulator (cl-sdk) and on a real CL1, so this
is the only place aware of the difference.
"""

from __future__ import annotations

import os

from ..config import NeuralCfg


def get_cl(cfg: NeuralCfg, accelerated: bool = False):
    os.environ.setdefault("CL_SDK_VISUALISATION", "0")
    if cfg.sdk_seed >= 0:
        os.environ["CL_SDK_RANDOM_SEED"] = str(cfg.sdk_seed)
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1" if accelerated else "0"
    import cl

    return cl


def backend_name(cl_mod) -> str:
    try:
        sim = cl_mod.is_simulator() if callable(cl_mod.is_simulator) else bool(cl_mod.is_simulator)
    except Exception:
        sim = True
    return "cl-sdk simulator" if sim else "CL1 hardware"
