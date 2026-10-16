"""Redraw every figure that comes from saved results alone (no GPU, no embeddings): error vs horizon per run,
per dataset and per seed, the near-miss test, the ranking test and the probe-scaling curve."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

for path in stages.redraw(args.root):
    print(path)
