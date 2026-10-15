"""Dataset adapters: each turns a raw source into episodes (see pwm.episodes)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pwm.datasets import pusht, toy
from pwm.episodes import Episode


@dataclass(frozen=True)
class Dataset:
    build: Callable[[Path], list[Episode]]  # raw folder (downloads, recordings) -> episodes
    step: float  # seconds between the frames the model sees
    move_threshold: float  # table units; see pwm.evaluator


DATASETS = {
    "toy": Dataset(build=toy.build, step=toy.STEP, move_threshold=toy.MOVE_THRESHOLD),
    "pusht": Dataset(build=pusht.build, step=pusht.STEP, move_threshold=pusht.MOVE_THRESHOLD),
}
