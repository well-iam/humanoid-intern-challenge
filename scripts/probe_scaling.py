"""Probe error against the number of training clips: the data-scaling figure.

Cross-validation over the training clips only (held-out clips are never read): for each of 5 folds and each size
n, the probe is fitted on n clips drawn from the other folds and scored on the fold. Writes probe-scaling.json and
probe-scaling.png to the run folder.

    uv run python scripts/probe_scaling.py --dataset clips --encoder dinov2
"""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--sizes", type=int, nargs="+", help="numbers of clips (default: 6, 12, 24, ... and all)")
args = parser.parse_args()

curve = stages.probe_scaling(args.dataset, args.encoder, args.root, sizes=args.sizes)
for point in curve["points"]:
    print(f"{point['clips']:4d} clips: {point['error']:.2f} {curve['unit']}")
