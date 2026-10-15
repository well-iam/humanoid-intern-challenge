"""Stage 3: train the predictor (or the action-blind predictor) on the cached frame embeddings."""

import argparse

from pwm import stages
from pwm.predictor import SIZES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--root", help="data and run root (default: $PWM_DATA, else ./data)")
parser.add_argument("--blind", action="store_true", help="train the action-blind predictor")
parser.add_argument("--size", choices=SIZES, default="real")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, help="default: the size's learning rate")
parser.add_argument("--device", help="default: cuda if available, else cpu")
parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="fp16: autocast + gradient scaler")
parser.add_argument("--seed", type=int, default=0, help="weights and batches; n > 0 saves to <encoder>/seed-n/")
args = parser.parse_args()

stages.train(
    args.dataset,
    args.encoder,
    args.root,
    blind=args.blind,
    device=args.device,
    size=args.size,
    steps=args.steps,
    batch_size=args.batch_size,
    lr=args.lr,
    precision=args.precision,
    seed=args.seed,
    log_every=100,
)
