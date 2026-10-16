"""Dataset adapters: each turns a raw source into episodes (see pwm.episodes)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pwm.datasets import pusht, toy, tracking
from pwm.episodes import Episode


@dataclass(frozen=True)
class Dataset:
    build: Callable[[Path], list[Episode]]  # raw folder (downloads, recordings) -> episodes
    step: float  # seconds between the frames the model sees
    move_threshold: float  # table units; see pwm.evaluator
    raw: str | None = None  # the raw folder, relative to the root; default <dataset>/raw


DATASETS = {
    "toy": Dataset(build=toy.build, step=toy.STEP, move_threshold=toy.MOVE_THRESHOLD),
    "pusht": Dataset(build=pusht.build, step=pusht.STEP, move_threshold=pusht.MOVE_THRESHOLD),
    # My takes: Friday's recordings go in <root>/clips/raw/.
    "clips": Dataset(build=tracking.build, step=tracking.STEP, move_threshold=tracking.MOVE_THRESHOLD),
    # The test takes: setup checks only, never project data, so a dataset of its own. Each take is tracked in its
    # own setup (pwm.datasets.setups.of_take): the morning one is the wallet on wood.
    "clips-test-take": Dataset(
        build=tracking.build, step=tracking.STEP, move_threshold=tracking.MOVE_THRESHOLD, raw="clips/test-take"
    ),
}
