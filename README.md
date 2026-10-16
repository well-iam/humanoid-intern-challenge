# Pushing world model

I filmed 115 short clips of my hand pushing a box on a table with a phone: 80 pushes, and 35 near-misses where my hand passes the box without touching it. I used them for two things:

1. **A world model.** A small model that predicts where the box will be, given how my hand is about to move. I tested the method on Push-T, a public pushing dataset, before recording anything.
2. **A robot replay.** A simulated Franka Panda arm follows my palm paths and pushes a simulated box. This checks that hand paths from a phone video work as robot paths.

The world model is the main part. It does not drive the robot. I tried to use it to choose pushes, but its favourite push was worse than a random one (see [Can the model choose a push?](#can-the-model-choose-a-push)).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/well-iam/humanoid-intern-challenge/blob/main/notebooks/pushing_world_model.ipynb) Rebuilds every figure and GIF from cached results, on the CPU, in about 12 min.

![My clip, the model's prediction, the Panda replay](assets/gif-strip.gif)

*Left:* one of my test clips, which the model never saw in training (palm in green).\
*Middle:* what the model expected this moment to look like, 1 s before it happened, given my palm's moves. A prediction is a feature vector, not an image, so I show the training frame whose features are closest to it.\
*Right:* the Panda replaying my palm path in MuJoCo.\
This is a typical clip, not the best. The simulated box ends 4.5 cm from where my real box ended, and turned 41° differently.

## Results in short

| | result |
|---|---|
| ✅ World model, my clips, 1 s ahead | box placed 2.82 cm off, against 4.16 cm for the same model without my hand's moves, and 6.67 cm for "the box stays still". Better on all 10 test pushes. |
| ❌ World model as a planner (Push-T) | its ranking of pushes is better than chance, but its top pick is worse than a random pick |
| ❌ Near-misses | the model still imagines a small push that never happened (improved, not solved) |
| ✅ Robot replay | after a push of about 15 cm, the simulated box typically ends 3.8 to 4.7 cm from where my real box ended. On all 35 near-misses, the arm passes the box without moving it. |
| ❌ Robot replay, box angle | typically 25° off |

**Caveats on the world-model result:** one training run, 18 test clips (10 pushes). 8 of those test clips were also used for a first run that failed. On Push-T, the same pass mark held on only 1 of 3 training runs.

## How it works

- **Encoder:** DINOv2-small, a ready-made image model, turns each frame into a 16 × 16 grid of feature vectors. I don't train it.
- **Predictor:** a small transformer (about 20M parameters), the only part I train. Its inputs are the last 3 frames' features, the pusher's position, and the action: how far the pusher moves over the next step. Its output is the change to the next frame's features. That change starts at zero, so an untrained predictor equals "nothing moves". The loss is L1 on the next frame plus L1 after a 2-step rollout. Feeding its own predictions back in gives predictions 1, 2, 4 and 8 steps ahead.
- **Probe:** a linear read-out, fitted on real frames, that reads the object's position from any features, real or predicted. So every score is a distance on the table, not a feature error.

Frozen DINOv2 features plus a trained predictor is close to DINO-WM (Zhou et al., 2024). What I add is the probe that scores in centimetres, the controls below, and my own data.

The action is the pusher's observed move, not a controller command. That is all hand tracking can give me, so I did the same on Push-T.

**Controls.** I compare the predictor with:

- **"Nothing moves"**: the object stays where it is.
- **Action-blind model**: the same model, trained the same way, that sees where the pusher is but not where it will move next. This is the hard one to beat.
- **Wrong actions**: the predictor fed my hand's moves reversed or shuffled. If it really uses the actions, it should get worse.

I only score moments where the object really moves. Otherwise "nothing moves" wins for free.

**Pass marks**, committed before any result:

1. **The probe works:** on real frames, it finds the object within 4% of the table's width (2 cm on my 50 cm table, 20 units on Push-T's 512-unit board).
2. **The predictor uses the actions:** 4 steps ahead, it is at least 20% better than both baselines, and wrong actions make it worse.

"Nothing moves" is always beaten by far, so below I report the margin over the action-blind model.

One step is 0.25 s on my clips and 0.5 s on Push-T, so "4 steps ahead" is 1 s on my clips and 2 s on Push-T.

## My data

### Recording

I filmed my hand pushing a white box (17.8 × 5.8 × 3 cm) on a carpet, with a phone held 71 cm above the table. I recorded 9 takes and clapped before each push, so the claps cut the takes into clips without me touching the phone. That gave 119 clips. I dropped 4 where my hand came back into view to reset the box, which left **115 clips: 80 pushes and 35 near-misses**. The near-misses are there on purpose: a good model should predict that the box stays still.

### From video to numbers

- **Table:** a printed ArUco marker lets me straighten each frame into a top-down view of a 50 × 50 cm square, at 8 frames per second. In this view, the A4 sheet under the marker measures within 0.28 cm of its real size, and the box's top within 0.77 cm.
- **Hand:** MediaPipe, a free hand tracker, finds my hand. The pusher position is the centre of my palm.
- **Box:** the bright rectangle of the right size.
- **End of a clip:** when my hand has been out of view for 0.75 s, so no clip includes my hand coming back.

### Train and test split

I kept **18 clips for testing**: 10 pushes and 8 near-misses. 8 of them are from my first 3 takes, chosen before I trained anything. The other 10 were drawn at random (fixed seed) from the later takes. The other **97 clips train** the models. The model and recipe are the same as on Push-T. Nothing was tuned on the test clips.

## Results on my clips

The probe reads the box from real test frames within 1.24 cm.

Error on all 18 test clips, in cm. Lower is better:

| | 1 s ahead | 2 s ahead |
|---|---:|---:|
| **the predictor** | **2.82** | **4.02** |
| action-blind model | 4.16 | 5.96 |
| "nothing moves" | 6.67 | 11.84 |
| predictor, my hand's moves reversed | 6.43 | 11.35 |
| predictor, my hand's moves shuffled | 5.06 | 8.83 |

At 1 s, the predictor is 32% better than the action-blind model, and it beats it on every one of the 10 test pushes. Given my hand's moves reversed, it is as bad as "nothing moves", so it really uses them. All from one training run.

![Error vs horizon on my clips](assets/clips-error-vs-horizon.png)

### Near-misses

![Near-miss test](assets/near-miss.png)

On the 8 test near-misses, 1 s ahead, the predictor moves the box 1.44 cm. On the real, still box, the probe already reads 0.67 cm of movement. So the model still imagines a small push that never happened. The action-blind model moves it 2.17 cm.

### The first try failed; more clips fixed it

My first run used only my first 3 takes: 22 training clips, tested on the same 8 test clips from those takes. It failed both pass marks. I recorded 6 more takes and reran everything unchanged. Because I added data after seeing the failure, the second run is a second look at those 8 test clips.

<details>
<summary>Error 1 s ahead on those 8 test clips, 22 vs 97 training clips</summary>

In cm. Lower is better:

| | 22 training clips | 97 training clips |
|---|---:|---:|
| **the predictor** | **3.78** | **3.15** |
| action-blind model | 4.46 | 4.61 |
| "nothing moves" | 5.83 | 6.34 |
| predictor, my hand's moves reversed | 5.29 | 6.45 |
| predictor, my hand's moves shuffled | 4.42 | 5.10 |

| pass mark | 22 training clips | 97 training clips |
|---|---|---|
| probe reads the box from real frames within 2 cm | fail: 3.49 cm | pass: 1.72 cm |
| predictor at least 20% better than the action-blind model | fail: 15% | pass: 32% |

- Only the training data changed.
- With 97 clips, reversing my hand's moves makes the predictor as bad as "nothing moves" (6.45 vs 6.34 cm). With 22, it was still better (5.29 vs 5.83). So it now relies more on my hand's moves.
- "Nothing moves" also changed (5.83 to 6.34 cm), because it is read by the probe too. My guess, not tested: the weaker first probe pulled positions toward the middle of the table, which hid part of the motion.
- The near-miss error also dropped: the first run moved the still box 2.37 cm, and did not move it less than the action-blind model.

</details>

**Why more clips helped: the probe.** With 22 training clips, the probe was 3.49 cm off even on real frames, so every score had that error built in. Feeding DINOv2 sharper frames (448 px instead of 224 px) did not help. More clips did:

![Probe error vs number of training clips](assets/probe-scaling.png)

I fitted the probe on more and more of the 97 training clips and tested it on the training clips it had not seen (5-fold cross-validation). The 18 test clips were never used. The probe gets under 2 cm somewhere between 24 and 48 clips. Clips from the same take can land on both sides of a split, which may flatter the curve a little. The test-clip numbers (1.72 and 1.24 cm) agree with it.

## Checking the method on Push-T

Before recording, I tested the method on Push-T: a round pusher moves a T-shaped block on a 512-unit board, and every position is recorded exactly. I trained on 100 recordings and kept 50 for testing. I also tried a much larger encoder, V-JEPA 2 ViT-L, with everything else the same.

Error 2 s ahead on the 50 test recordings, in board units (only where the T moves). Lower is better:

| | DINOv2-small | V-JEPA 2 |
|---|---:|---:|
| **the predictor** | **18.14** | **18.96** |
| action-blind model | 24.08 | 22.80 |
| "nothing moves" | 44.55 | 44.44 |
| predictor, pusher's moves reversed | 46.64 | 46.64 |
| predictor, pusher's moves shuffled | 38.71 | 37.84 |

| pass mark | DINOv2-small | V-JEPA 2 |
|---|---|---|
| probe reads the T from real frames within 20 units | pass: 3.96 | pass: 8.27 |
| predictor at least 20% better than the action-blind model | pass: 25% | fail: 17% |

With DINOv2, the predictor is off by 18 units, about 3.5% of the board. With the moves reversed, it is as bad as "nothing moves", so it really uses them.

![Error vs horizon on Push-T, both encoders](assets/pusht-error-vs-horizon.png)

**Was the pass luck? Partly.** I trained the DINOv2 predictor and its action-blind model twice more, with new random seeds. The 20% mark held on 1 of 3:

| seed | predictor | action-blind | needed (under 80% of action-blind) | result |
|---|---:|---:|---:|---|
| 0 (above) | 18.14 | 24.08 | 19.26 | pass |
| 1 | 17.95 | 21.31 | 17.05 | fail, by 0.90 |
| 2 | 35.19 | 43.46 | 34.77 | fail: training got stuck, both barely beat "nothing moves" |

The predictor is about the same on seeds 0 and 1. The action-blind model trained better on seed 1 and raised the bar. Per-seed plots: [assets/pusht-seeds.png](assets/pusht-seeds.png).

So this pass mark does not separate the two encoders. V-JEPA 2 missed it by 0.72 units, less than DINOv2's own seed 1, and I trained V-JEPA 2 only once. I kept DINOv2 for two reasons that don't depend on the seed: the probe reads the T about twice as precisely from its features (3.96 vs 8.27 units), and its cached features are smaller (3.5 vs 9.2 GB).

## Can the model choose a push?

A world model is useful for planning if it can tell a good action from a bad one before trying it. Push-T has a goal (a target place for the T) and its own simulator, so I tested this there, with the setup fixed before running it.

I took 50 moments from the test recordings. For each, I made 32 pusher paths, 4 s long: the path the human really took, plus 31 variations of it (turned, longer or shorter, bent). The model imagined each path and ranked them by how close it predicted the T would end to the goal. Push-T's simulator played every path for real, which gives the true ranking.

![Ranking test](assets/ranking.png)

How far the T really ended from the goal, in board units, averaged over the 50 moments:

| path | distance to the goal |
|---|---:|
| the model's top pick | 79.8 |
| a random pick, on average | 73.8 |
| the path the human really took | 47.6 |
| the best of the 32 paths | 31.6 |

- **The ranking as a whole is better than chance.** Its rank correlation with the simulator's ranking (1 is the same order, 0 is unrelated) is +0.26 on average, and above 0 in 36 of the 50 moments. With the pusher's moves shuffled, it drops to 0.00, so the ranking comes from the moves.
- **Its top pick is worse than a random pick** (79.8 vs 73.8).

**Why.** For its top picks, the model predicted the T would end 20.7 units from the goal. The simulator said 79.8. On those paths, the model's predicted end position was 69 units from where the T really ended, against 42 on an average path. So picking the path that looks best mostly picks the model's biggest mistakes. This is why I did not build a planner on it.

## Replaying my pushes on a Panda

MediaPipe gives the centre of my palm, but the part of my hand that touches the box is a different point. So at the moment of contact, I measure the gap from my palm to the nearest edge of the box, and shift the whole palm path by it. The Panda's closed gripper follows the shifted path 1.5 cm above the table in MuJoCo, using the mink library for inverse kinematics. The simulated box starts where my real box started. Nothing here is trained.

In 32 of the 80 pushes, MediaPipe found my hand less than 0.25 s before the box moved, so I don't know where my palm was at contact. For those, I use the typical shift from the other 48 pushes, turned to the direction my hand moves.

![Panda replaying one of my pushes](assets/panda-push.gif)

A test push where my hand was found late, so the path uses the typical shift. It ends 5.3 cm off: of the 7 such test pushes, 3 end closer and 3 further.

How far the simulated box ends from where my real box ended, on all 115 clips ("typical" is the median):

| clips | typical error | worst error | how far my real box moved, typically |
|---|---:|---:|---:|
| 48 pushes, with their own shift | 3.8 cm | 11.1 cm | 14.4 cm |
| 32 pushes where my hand was found late | 4.7 cm | 19.1 cm | 16.4 cm |
| 35 near-misses | 0.0 cm (the simulated box never moved) | 0.5 cm | 0.0 cm |

For scale: a simulated box that never moved would be off by exactly how far the real box moved, the last column. The near-miss row is a sanity check: the arm passes the box without moving it.

More clips helped here too. Among my first 30 clips, the late pushes' typical error dropped from 5.5 to 3.0 cm once the typical shift came from 48 pushes instead of 12.

Near-miss replay: [assets/panda-near-miss.gif](assets/panda-near-miss.gif). The same replay code also runs on Push-T paths: [assets/pusht-panda.gif](assets/pusht-panda.gif).

## Design choices

- **Predict features, score in centimetres.** I never decode predictions to pixels. The probe turns any prediction into a position, so I can compare models in physical units against simple baselines.
- **Push-T first.** It has exact positions and its own simulator, so I could check the method before recording.
- **Pass marks fixed before results.** The pass marks, the training recipe and the ranking test were committed before I ran them.
- **The full 16 × 16 grid, no pooling.** On a 50 cm square, a 4 × 4 pooled cell would cover 12.5 cm, and the 5.8 cm wide box would be a small part of it.
- **Near-misses in the data.** Without them, a model could learn "the hand moves, so the box moves" and never be caught.

What I left out:

| option | why |
|---|---|
| LIBERO | One of its old dependencies would not build with current tools. Plain MuJoCo with a ready-made Panda model installed in a minute. |
| SmolVLA fine-tuning | 4 to 5 h on a big GPU, about 5 times longer on a free one. I spent the time on the world model. |
| Planning with the model | The ranking test showed a planner would pick the model's mistakes. |
| Predicting the box's angle | Out of time. The probe reads position only. |
| Decoding predictions to pixels | A second model to train. The nearest real frame is enough to show what a prediction means. |

## What didn't work

- **One training run on my clips.** On Push-T, the 20% mark held on 1 of 3 runs, and on one run training got stuck; I did not find out why. My clips result may not hold on another seed.
- **Near-misses:** the model still imagines a small push when my hand only passes by (1.44 cm, against the 0.67 cm the probe reads on a still box).
- **Choosing a push:** the model's top pick is worse than a random one.
- **Hands found late:** in 32 of 80 pushes, the robot replay has to guess the contact shift and does worse (worst case 19.1 cm). Of my 10 test pushes, only 3 have their own shift.
- **Box angle in the robot replay:** typically 25° off. The Panda pushes with two closed fingers (3.8 × 2 cm). My hand pushes with a wide edge, so the box turns differently.
- **Box size on the new takes:** the box measured within 0.33 cm of its real size on the first 3 takes, but up to 0.77 cm off on all 9. I did not find out why.
- **Push-T paths on the Panda:** most end close (typically about 2 cm off at the Panda's scale), but the worst is far off. Push-T's simulator has no floor friction and MuJoCo does, so the T drifts differently, and the human's later corrections then push it the wrong way.
- **One setup only:** one box, one carpet, one camera height, one hand.

## How to run

### Colab

Open the notebook with the badge at the top, then Runtime > Run all. By default it downloads the cached features, trained predictors and tracks from the Hugging Face dataset [`well-iam/humanoid-intern-challenge`](https://huggingface.co/datasets/well-iam/humanoid-intern-challenge), and rebuilds every figure and GIF on the CPU. This took about 12 min on a fresh Colab runtime.

To retrain, set `RETRAIN = "clips"` (my clips) or `"all"` (also Push-T with DINOv2). I have not timed these on Colab. My estimate is about 40 min and 1.5 h on a T4. Push-T is downloaded from its original URL, never re-hosted.

### Locally

With [uv](https://docs.astral.sh/uv/), Python 3.12 and an NVIDIA GPU (I used a Quadro RTX 4000, 8 GB):

```bash
uv sync
export PWM_DATA=/path/to/pwm-data   # every stage reads and writes here
uv run pytest                       # fast tests, on a toy world
uv run pytest -m slow               # data, simulator and long-training tests
```

To get all the cached results (9.6 GB):

```bash
uv run hf download well-iam/humanoid-intern-challenge --repo-type dataset --local-dir $PWM_DATA
```

To rerun every stage, from raw data to the figures (**not run end to end**: I wrote this script from the commands I ran, but did not run it in one go before the deadline):

```bash
bash scripts/reproduce.sh pusht   # Push-T: both encoders, extra seeds, ranking test, Panda replay
bash scripts/reproduce.sh clips   # my clips
bash scripts/reproduce.sh all     # both, then copy the figures into assets/
```

Each model trains in about 15 min. Each step is its own script in `scripts/`, so you can rerun one alone. The 1080p takes of my clips are not public. If `$PWM_DATA/clips/raw/` is empty, the script downloads the tracked clips from Hugging Face and starts from there.

## License

Code: MIT (see `LICENSE`). My clips and the cached results on Hugging Face: CC BY 4.0.
