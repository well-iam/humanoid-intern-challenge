#!/usr/bin/env bash
# Rerun every stage, from raw data to the README's figures. Needs an NVIDIA GPU and PWM_DATA set.
#
#   bash scripts/reproduce.sh pusht   # Push-T: both encoders, the extra seeds, ranking test, Panda replay
#   bash scripts/reproduce.sh clips   # my clips: tracking (or the tracked episodes from Hugging Face) onwards
#   bash scripts/reproduce.sh all     # both, then copy the figures into assets/
#
# Each model trains in about 15 min on an 8 GB GPU. The script stops at the first error.

set -euo pipefail
: "${PWM_DATA:?set PWM_DATA to the data folder: every stage reads and writes there}"

py() { uv run python "$@"; }
TRAIN=(--precision fp16 --batch-size 16 --steps 2500 --lr 1e-3)
FIRST_TAKES=(VID_20261016_141304 VID_20261016_141455 VID_20261016_141631)  # the first run's 8 test clips

pusht() {
  py scripts/data.py --dataset pusht
  for enc in dinov2 vjepa2; do
    py scripts/embed.py --dataset pusht --encoder "$enc"
    py scripts/train.py --dataset pusht --encoder "$enc" "${TRAIN[@]}"
    py scripts/train.py --dataset pusht --encoder "$enc" "${TRAIN[@]}" --blind
    py scripts/evaluate.py --dataset pusht --encoder "$enc"
  done
  py scripts/compare.py --dataset pusht  # error-vs-horizon.png, winner.json
  for seed in 1 2; do
    py scripts/train.py --dataset pusht --encoder dinov2 "${TRAIN[@]}" --seed "$seed"
    py scripts/train.py --dataset pusht --encoder dinov2 "${TRAIN[@]}" --blind --seed "$seed"
    py scripts/evaluate.py --dataset pusht --encoder dinov2 --seed "$seed"
  done
  py scripts/seeds.py --dataset pusht --encoder dinov2 --seeds 0 1 2
  py scripts/rank.py --dataset pusht --encoder dinov2
  py scripts/panda_replay.py --dataset pusht
}

clips() {
  if compgen -G "$PWM_DATA/clips/raw/*.mp4" > /dev/null; then
    py scripts/track.py
  else  # the 1080p takes are not public: start from the tracked episodes on Hugging Face
    uv run hf download well-iam/humanoid-intern-challenge --repo-type dataset --local-dir "$PWM_DATA" \
      --include "clips/*" --exclude "clips/videos/*"
  fi
  py scripts/embed.py --dataset clips --encoder dinov2
  py scripts/train.py --dataset clips --encoder dinov2 "${TRAIN[@]}"
  py scripts/train.py --dataset clips --encoder dinov2 "${TRAIN[@]}" --blind
  py scripts/evaluate.py --dataset clips --encoder dinov2
  py scripts/near_miss.py --dataset clips --encoder dinov2
  py scripts/probe_scaling.py --dataset clips --encoder dinov2
  # the same model graded on the first run's 8 test clips only (the before/after table)
  py scripts/evaluate.py --dataset clips --encoder dinov2 --only-takes "${FIRST_TAKES[@]}"
  py scripts/near_miss.py --dataset clips --encoder dinov2 --only-takes "${FIRST_TAKES[@]}"
  py scripts/panda_replay.py --dataset clips --episode VID_20261016_163841-02 VID_20261016_165020-05
  py scripts/gif_strip.py --episode VID_20261016_141304-11
}

case "${1:-}" in
  pusht) pusht ;;
  clips) clips ;;
  all) pusht; clips; py scripts/readme_assets.py ;;
  *) echo "usage: bash scripts/reproduce.sh pusht|clips|all" >&2; exit 2 ;;
esac
