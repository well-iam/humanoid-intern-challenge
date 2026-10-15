"""Stage: the ranking test on Push-T, judged by Push-T's own simulator.

Run it only if the validation gate passed. Writes <root>/pusht/<encoder>/ranking/.
"""

import argparse

from pwm import stages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="pusht")
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--candidates", type=int, default=32, help="candidate paths per starting state")
parser.add_argument("--horizon", type=int, default=8, help="steps each candidate path lasts")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

report = stages.rank(args.dataset, args.encoder, root=args.root, n_candidates=args.candidates,
                     horizon=args.horizon, seed=args.seed)
print(report.to_markdown())
