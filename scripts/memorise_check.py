"""Bug check before a full training run: the predictor must memorise one training episode.

Trains a predictor on the training episode whose object travels furthest, then rolls it out
on that same episode. The probe is fitted on the other training episodes. A predictor that
works gets close to the probe floor at every horizon; one with a bug stays near the static
baseline.
"""

import argparse
import dataclasses
import time

import numpy as np

from pwm import stages
from pwm.datasets import DATASETS
from pwm.episodes import load_episodes
from pwm.evaluator import evaluate
from pwm.predictor import SIZES, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--size", choices=SIZES, default="real")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, help="default: the size's learning rate")
parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
args = parser.parse_args()

episodes = load_episodes(stages.data_root(args.root) / args.dataset / "episodes")
train_episodes = [ep for ep in episodes if ep.split == "train"]
one = max(train_episodes, key=lambda ep: np.linalg.norm(np.diff(ep.object_pos, axis=0), axis=1).sum())
check = [ep for ep in train_episodes if ep is not one] + [dataclasses.replace(one, split="held-out")]
embeddings = stages.load_embeddings(args.dataset, args.encoder, check, args.root)
spec = DATASETS[args.dataset]
print(f"memorising {one.name} ({len(one.frames)} frames)", flush=True)

start = time.time()
predictor = train(
    [one],
    embeddings[-1:],
    spec.step,
    size=args.size,
    steps=args.steps,
    batch_size=args.batch_size,
    lr=args.lr,
    device=stages._device(),
    precision=args.precision,
    log_every=50,
)
print(f"trained in {time.time() - start:.0f} s", flush=True)

report = evaluate(check, embeddings, spec.step, spec.move_threshold, predictor=predictor.rollout)
print(report.to_markdown(f"memorise {one.name}, {args.dataset}, {args.encoder} encoder"))
