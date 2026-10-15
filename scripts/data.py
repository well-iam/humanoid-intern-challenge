"""Stage 1: turn a dataset's raw source into episodes in the run folder."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

stages.make_episodes(args.dataset, args.root)
