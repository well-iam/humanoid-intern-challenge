"""Compare the evaluated runs of one encoder trained with different seeds (train and evaluate
with --seed first): writes seeds.md and error-vs-horizon-seeds.png into the encoder folder."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

print(stages.seeds(args.dataset, args.encoder, args.seeds, args.root).read_text(encoding="utf-8"))
