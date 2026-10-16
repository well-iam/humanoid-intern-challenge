"""Stage 4b: the near-miss test. How far the trained predictors move the object on held-out windows where
the pusher passes it by, next to windows where it really moves."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--only-takes", nargs="+", help="score only the held-out clips of these takes (results in <run>/only-takes/)")
args = parser.parse_args()

stages.near_miss(args.dataset, args.encoder, args.root, only_takes=args.only_takes)
print((stages.run_folder(args.dataset, args.encoder, args.root) / ("only-takes" if args.only_takes else "") / "near-miss.md").read_text())
