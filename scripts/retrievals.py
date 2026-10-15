"""Nearest-real-frame retrievals for a few held-out rollouts (default encoder: the winner)."""

import argparse
import json

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", help="default: the winning encoder in winner.json")
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--rollouts", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

folder = stages.data_root(args.root) / args.dataset
encoder = args.encoder or json.loads((folder / "winner.json").read_text())["encoder"]
print(stages.retrievals(args.dataset, encoder, args.root, n_rollouts=args.rollouts, seed=args.seed))
