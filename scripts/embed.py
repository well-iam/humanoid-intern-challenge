"""Stage 2: compute and cache the frame embeddings of every episode with a frozen encoder."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
args = parser.parse_args()

stages.embed(args.dataset, args.encoder, args.root)
