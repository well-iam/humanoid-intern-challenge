# Pushing world model

I trained a small world model that predicts where a pushed object will go, checked it on Push-T, ran it on phone clips of my own hand pushing a box, and replayed my palm paths on a simulated Franka Panda.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/well-iam/humanoid-intern-challenge/blob/main/notebooks/pushing_world_model.ipynb)

<!-- UPDATE: gif-strip.gif is redrawn by scripts/gif_strip.py after the new run. Its default clip is the first held-out push with its own contact shift; the held-out split changes with more takes, so the clip name in this caption may change too. -->
![My clip, the model's prediction, the Panda replay](assets/gif-strip.gif)

Left: one of my held-out clips (palm in green). Middle: what the predictor expected 1 s earlier, from the frames it had then and my real palm moves. A prediction is a feature vector, not an image, so I show the training frame whose features are closest to it. Right: the Panda replaying my palm path in MuJoCo.

## The idea

- A frozen image encoder (DINOv2-small) turns each frame into a 16 × 16 grid of features.
- A small predictor (6 transformer layers, about 20M parameters) is the only part I train. It takes the last 3 frames' features, where the pusher is, and how it moves next. It outputs the next frame's features. Feeding its output back in gives predictions 1, 2, 4 and 8 steps ahead.
- A linear probe reads the object's position out of real or predicted features. So every score is a distance on the table, not a loss on feature vectors.

Each predictor is compared with two baselines: the **static baseline** (nothing moves) and an **action-blind predictor** (same model and training, but it does not see how the pusher moves). The **wrong-action test** reruns the predictor with reversed or shuffled pusher moves. I score only **moving windows**, where the object really moves within the horizon, because on still windows "nothing moves" wins for free.

I fixed two pass marks before seeing any result:

- **Gate 1:** the probe finds the object on real frames within 4% of the workspace width.
- **Validation gate:** at 4 steps ahead, the predictor's error is at least 20% below the better baseline, and wrong pusher moves make it worse.

## Push-T results

Push-T is a public dataset of a round pusher moving a T-shaped block, with exact positions. I used 100 episodes for training and 50 held out, with a step of 0.5 s. The workspace is 512 units wide.

![Error vs horizon on Push-T, both encoders](assets/pusht-error-vs-horizon.png)

Error at 4 steps (2 s) ahead, held-out moving windows, in Push-T units:

| | DINOv2-small | V-JEPA 2 ViT-L |
|---|---:|---:|
| probe on real frames (Gate 1: under 20) | 3.96 | 8.27 |
| predictor | **18.14** | **18.96** |
| action-blind predictor | 24.08 | 22.80 |
| static baseline | 44.55 | 44.44 |
| reversed pusher moves | 46.64 | 46.64 |
| shuffled pusher moves | 38.71 | 37.84 |
| validation gate (pass mark) | pass (19.26) | fail (18.24) |

The predictor is 59% below the static baseline. Reversed pusher moves are as bad as predicting no motion at all. So the predictor uses the actions. DINOv2 won and is the only encoder I used after this.

**The DINOv2 pass held on only 1 of 3 seeds.** I retrained both DINOv2 models with seeds 1 and 2 and the same recipe:

| seed | predictor | action-blind | pass mark | gate |
|---|---:|---:|---:|---|
| 0 | 18.14 | 24.08 | 19.26 | pass by 1.12 |
| 1 | 17.95 | 21.31 | 17.05 | fail by 0.90 |
| 2 | 35.19 | 43.46 | 34.77 | fail: training got stuck |

The predictor itself is stable on the two healthy seeds (18.14 and 17.95). The action-blind predictor is not, and the gate is measured against it. The seed-0 pass came partly from a weak action-blind draw. Per-seed plots: [assets/pusht-seeds.png](assets/pusht-seeds.png).

## Ranking test: can the model score paths?

A world model is useful if it can tell good actions from bad ones. For each of 50 held-out starting states, I made 32 candidate pusher paths of 4 s by rotating, scaling and bending the recorded one. The predictor ranked them by how close it expected the T to end up to Push-T's goal. Then Push-T's own simulator played every path and gave the true answer.

![Ranking test](assets/ranking.png)

- The two rankings agree: mean Spearman rank correlation per start **+0.26 ± 0.06**, positive in 36 of 50 starts. With shuffled pusher moves it is **+0.00 ± 0.03**.
- **But the single top pick is worse than a random pick:** it ends 79.8 units from the goal, against 68.3 for a random pick (73.8 on average over all candidates).

Why: the top pick is where the model is most wrong in the hopeful direction. For top picks it predicted 20.7 units from the goal; the simulator said 79.8. Its position error on top picks is 69 units, against 42 on average. Taking the best-scored path selects the model's own mistakes. That is the usual way a planner exploits a learned model. After the test (not part of the fixed setup), I checked that the top 5, 8 and 16 picks do beat random (71.7, 69.1, 67.5 vs 73.8). The ranking has signal in bulk, but its best guess is not reliable.

## My data

<!-- UPDATE: every number and figure in this section comes from the first 30 clips (ticket 14 and 15). The new run with about 120 clips replaces them; the 30-clip run stays in pwm-data/clips/dinov2-30clips/. -->

I recorded <!-- UPDATE: clip count, takes -->30 clips in 3 takes<!-- /UPDATE -->: my hand pushing a white box (17.8 × 5.8 × 3 cm) on a carpet, filmed from above by a phone 71 cm up. <!-- UPDATE: push / near-miss / held-out counts -->22 are pushes and 8 are near-misses, where my hand passes the box without touching it. 8 clips (6 pushes, 2 near-misses) are held out.<!-- /UPDATE --> Each clip starts at a clap, so I never touched the phone during a take.

From video to data:
- A printed ArUco marker gives the table plane. Each frame is straightened to a 50 × 50 cm square, at 8 frames per second.
- MediaPipe finds the hand; the pusher position is the centre of the palm.
- The box is the bright rectangle of the right size. Its measured top face is within <!-- UPDATE: tracking report -->0.33 cm<!-- /UPDATE --> of the ruler size, so the pixel-to-cm error is a few mm.

Same chain and training recipe as Push-T, with a step of 0.25 s. Nothing was changed after seeing held-out numbers.

<!-- UPDATE: assets/clips-error-vs-horizon.png from pwm-data/clips/dinov2/error-vs-horizon.png -->
![Error vs horizon on my clips](assets/clips-error-vs-horizon.png)

Error on held-out moving windows, cm:

<!-- UPDATE: fill the ~120 clips column from pwm-data/clips/dinov2/report.md; the 30 clips column comes from pwm-data/clips/dinov2-30clips/report.md -->
| | 30 clips, 1 s | 30 clips, 2 s | ~120 clips, 1 s | ~120 clips, 2 s |
|---|---:|---:|---:|---:|
| probe on real frames (Gate 1: under 2 cm) | 3.49 | | TBD | |
| predictor | **3.78** | **4.96** | TBD | TBD |
| action-blind predictor | 4.46 | 6.02 | TBD | TBD |
| static baseline | 5.83 | 10.12 | TBD | TBD |
| reversed pusher moves | 5.29 | 8.98 | TBD | TBD |
| shuffled pusher moves | 4.42 | 7.41 | TBD | TBD |
| validation gate (pass mark) | fail (3.57) | | TBD | |

<!-- UPDATE: all numbers in the next three bullets -->
- The predictor halves the static baseline's error at 2 s (4.96 vs 10.12 cm), and beats it at 1 s in all 6 held-out push clips. Wrong pusher moves make it worse.
- It is only 15% below the action-blind predictor, and the gate asks for 20%. Resampling the 6 held-out push clips puts the ratio at 0.85 (95% interval 0.78 to 0.92).
- **Gate 1 fails:** the probe is off by 3.49 cm on real frames. This error sits under every number above.

The probe got better with every clip I added. Cross-validated on the training clips: 6.27 cm with 6 clips, 4.86 with 9, 3.71 with 12, 3.38 with 15, 3.17 with 17. Feeding DINOv2 448 px frames instead of 224 px did not help (3.15 vs 3.04 cm). So I recorded more clips.

<!-- UPDATE: add assets/probe-vs-clips.png (probe error vs number of training clips, 30-clip and ~120-clip runs) once the new run draws it, and add it to ASSETS in scripts/readme_assets.py, then uncomment the line below (left commented so GitHub shows no broken image)
![Probe error vs number of training clips](assets/probe-vs-clips.png)
-->


**Near-miss test.** When my hand passes the box without touching it, the box should not move.

<!-- UPDATE: assets/near-miss.png and the numbers below -->
![Near-miss test](assets/near-miss.png)

At 1 s, the predictor moves the box 2.37 cm on near-miss windows. The probe's own jitter on the real frames is 1.60 cm, and the action-blind predictor moves it 1.87 cm. So the model learned "the box moves when the hand moves", not "the box moves when the hand touches it". This rests on 2 held-out near-miss clips.

### My palm paths drive the Panda

The palm centre is about 12 cm behind the part of the hand that touches the box. So for each push I measure, at the moment of contact, the vector from the palm to the nearest edge of the box, and move the whole path by it. The Panda's closed gripper follows the moved path 1.5 cm above the table, with mink inverse kinematics in MuJoCo. The simulated box starts where the real one started.

<!-- UPDATE: panda-push.gif is clip VID_20261016_141455-07 (the own-shift push with the median error); pick again from the new replay.json -->
![Panda replaying one of my pushes](assets/panda-push.gif)

Final box position, simulation vs my clip:

<!-- UPDATE: all rows from pwm-data/clips/panda/replay.json -->
| clips | how the path is moved | error, median (worst) | real box moved, median |
|---|---|---:|---:|
| 12 pushes | its own contact shift | 3.6 cm (5.9) | 14.8 cm |
| 10 pushes where the hand was found late | the median shift, turned to the hand's direction | 5.5 cm (22.7) | 15.7 cm |
| 8 near-misses | the median shift | box moved 0.0 cm in all 8 | 0.0 cm |

<!-- UPDATE: panda-near-miss.gif is clip VID_20261016_141631-03 (first held-out near-miss) -->
The near-misses stay near-misses on the robot: [assets/panda-near-miss.gif](assets/panda-near-miss.gif). The same replay code runs on Push-T paths: [assets/pusht-panda.gif](assets/pusht-panda.gif).

## Design choices

- **Prediction in feature space, scored in centimetres.** I never decode predictions to pixels. The linear probe turns any prediction into a position, so I can compare models in physical units against simple baselines.
- **Push-T first.** It has exact positions and its own simulator, so I could check the method before recording. My data got a protected slot.
- **Pass marks fixed before results.** The gates, the training recipe and the ranking test setup were all committed before I ran them.
- **The pusher action is the observed move of the pusher, not the controller's command.** That is all hand tracking can give on my clips, so I used the same on Push-T.
- **DINOv2-small over V-JEPA 2 ViT-L.** I picked the winner by the held-out error at 2 s. DINOv2 was also twice as precise for the probe and 2.7 times smaller on disk (3.5 vs 9.2 GB of features).
- **The full 16 × 16 grid, no pooling.** On a 50 cm square, a 4 × 4 pooled cell would cover 12.5 cm, and the 5.8 cm wide box would be a small part of it.

Dropped, and why:

| option | why I dropped it |
|---|---|
| LIBERO | It pins robosuite 1.4.0 and mujoco below 3.9, and one of its dependencies does not build with CMake 4. Plain MuJoCo with the Menagerie Panda and mink installed in a minute. |
| SmolVLA fine-tuning | About 4 to 5 h for 20k steps on an A100, and a free T4 is about 5 times slower. My hand clips also have no robot actions. |
| Planning with the model (CEM) | 4 to 8 h more work, and with a predictor trained on so little data a planner tends to exploit its errors. The ranking test asks the same question for much less, and showed that effect. |
| The released V-JEPA 2-AC predictor | Built for ViT-g, not ViT-L. |
| Predicting the object's angle | Out of time. The probe reads position only. |
| Decoding predictions to pixels | A second model to train. The nearest real frame is enough to show what a prediction means. |

## What didn't work

- **The Push-T gate on more seeds:** passed on 1 of 3. Seed 1 missed by 0.90 units because its action-blind predictor was stronger. Seed 2 got stuck in training (both models close to the static baseline). I did not investigate seed 2 further.
- **V-JEPA 2:** failed the gate (16.8% below its action-blind predictor, 20% needed). Its probe error is twice DINOv2's (8.27 vs 3.96), so part of its error is in the read-out, not the prediction.
- **The ranking test's top pick:** worse than random (79.8 vs 73.8 units from the goal on average).
- <!-- UPDATE: own-data numbers --> **Gate 1 on my clips:** 3.49 cm vs 2 cm. Per the plan, a Gate 1 fail means "fix the input first". I trained anyway, for lack of time, and I report every number with this in mind.
- <!-- UPDATE --> **Contact on my clips:** the near-miss test failed (2.37 cm of predicted motion where the box stayed still).
- <!-- UPDATE --> **Late hands:** in 10 of 22 pushes MediaPipe found my hand less than 0.25 s before the box moved. Without the palm at contact, the replay has to guess the contact shift (median 5.5 cm error, worst 22.7).
- <!-- UPDATE --> **The box's angle in replay:** median error 25°. The Panda pushes with two closed fingers, 3.8 × 2 cm. My hand pushes with a wide edge, so the real box turns differently.
- **Open-loop Push-T replay on the Panda:** median 20 units (about 2 cm at the Panda's scale), worst 281 units. Push-T's 2D physics has no floor friction; MuJoCo does. Once the T drifts, the human's later corrections push it the wrong way.

## How to run

### Colab (no GPU needed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/well-iam/humanoid-intern-challenge/blob/main/notebooks/pushing_world_model.ipynb)

Runtime > Run all. By default the notebook downloads the cached features, trained predictors and tracks from the Hugging Face dataset [`<HF_REPO_ID>`](https://huggingface.co/datasets/<HF_REPO_ID>), and rebuilds every figure and GIF on the CPU. It took 11 min on my machine with the GPU hidden. Set `RETRAIN = "clips"` to retrain the predictors on my clips (about 40 min on a T4, estimated) or `"all"` to also redo Push-T with DINOv2 (about 1.5 h, estimated). Push-T is downloaded from its original URL, never re-hosted.

<!-- TODO: after the public upload, run the notebook once on a fresh Colab runtime and write the time here. -->
Tested Colab runtime: Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4.

### Locally

With [uv](https://docs.astral.sh/uv/), Python 3.12 and an NVIDIA GPU (I used a Quadro RTX 4000, 8 GB):

```bash
uv sync
export PWM_DATA=/path/to/pwm-data   # every stage reads and writes here
uv run pytest                       # fast tests, on a toy world
```

Push-T, end to end (each model trains in about 15 min):

```bash
uv run python scripts/data.py --dataset pusht
uv run python scripts/embed.py --dataset pusht --encoder dinov2
uv run python scripts/train.py --dataset pusht --encoder dinov2 --precision fp16 --batch-size 16 --steps 2500 --lr 1e-3
uv run python scripts/train.py --dataset pusht --encoder dinov2 --precision fp16 --batch-size 16 --steps 2500 --lr 1e-3 --blind
uv run python scripts/evaluate.py --dataset pusht --encoder dinov2
uv run python scripts/compare.py --dataset pusht
uv run python scripts/rank.py --dataset pusht --encoder dinov2
uv run python scripts/panda_replay.py --dataset pusht
```

My clips: put the takes in `$PWM_DATA/clips/raw/`, then

```bash
uv run python scripts/track.py
uv run python scripts/embed.py --dataset clips --encoder dinov2
uv run python scripts/train.py --dataset clips --encoder dinov2 --precision fp16 --batch-size 16 --steps 2500 --lr 1e-3
uv run python scripts/train.py --dataset clips --encoder dinov2 --precision fp16 --batch-size 16 --steps 2500 --lr 1e-3 --blind
uv run python scripts/evaluate.py --dataset clips --encoder dinov2
uv run python scripts/near_miss.py --dataset clips --encoder dinov2
uv run python scripts/panda_replay.py --dataset clips
uv run python scripts/gif_strip.py
uv run python scripts/readme_assets.py   # copies the README's figures into assets/
```

The 1080p takes are not public. To run the clip chain, download the Hugging Face dataset above into `$PWM_DATA`: it holds the tracked episodes, so you can skip `track.py` and start from `embed.py`.
