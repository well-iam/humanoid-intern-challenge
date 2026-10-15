"""Stage 4: fit the probe, score every model on held-out windows and write the report."""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--seed", type=int, default=0, help="evaluate the run trained with this seed (<encoder>/seed-n/)")
args = parser.parse_args()

report = stages.evaluate(args.dataset, args.encoder, args.root, seed=args.seed)
print((stages.run_folder(args.dataset, args.encoder, args.root, args.seed) / "report.md").read_text())
