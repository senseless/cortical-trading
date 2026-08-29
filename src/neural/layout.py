"""Electrode layout: which channels carry which meaning.

Sensory groups receive stimulation (place coding = which group, rate coding =
stim frequency). Motor regions are read for decoding. Overlap is validated at
config load; this class just provides convenient views.
"""

from __future__ import annotations

from ..config import NeuralCfg


class ElectrodeLayout:
    def __init__(self, cfg: NeuralCfg):
        self.channels = cfg.channels
        self.sensory: dict[str, list[int]] = {k: list(v) for k, v in cfg.sensory.items()}
        self.motor: dict[str, list[int]] = {k: list(v) for k, v in cfg.motor.items()}
        if not self.motor:
            raise ValueError("no motor regions configured (neural.layout.motor)")

    @property
    def all_sensory(self) -> list[int]:
        return sorted({ch for chans in self.sensory.values() for ch in chans})

    @property
    def all_motor(self) -> list[int]:
        return sorted({ch for chans in self.motor.values() for ch in chans})

    @property
    def all_assigned(self) -> list[int]:
        return sorted(set(self.all_sensory) | set(self.all_motor))

    def group(self, name: str) -> list[int]:
        if name in self.sensory:
            return self.sensory[name]
        if name in self.motor:
            return self.motor[name]
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return name in self.sensory or name in self.motor
