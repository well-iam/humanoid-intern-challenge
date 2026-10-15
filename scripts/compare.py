"""Stage 5: compare the evaluated encoders, plot error vs horizon and pick the winning encoder."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

winner = stages.compare(args.dataset, args.root)
print((stages.data_root(args.root) / args.dataset / "winner.json").read_text())
