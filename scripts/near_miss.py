"""Stage 4b: the near-miss test. How far the trained predictors move the object on held-out windows where
the pusher passes it by, next to windows where it really moves."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

stages.near_miss(args.dataset, args.encoder, args.root)
print((stages.run_folder(args.dataset, args.encoder, args.root) / "near-miss.md").read_text())
