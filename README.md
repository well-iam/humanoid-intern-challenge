# Pushing world model

I trained a small world model that predicts where a pushed object will go, checked it on Push-T, ran it on phone clips of my own hand pushing a box, and replayed my palm paths on a simulated Franka Panda.

My data plays two roles. It trains the world model, which predicts how the box will move given my hand's moves. And my palm paths, shifted to the point of contact, become the robot's paths. The world model does not steer the robot: when I asked it to choose between pushes on Push-T, its favourite was worse than a random one (see the [ranking test](#ranking-test-can-the-model-choose-a-good-push)), so I did not build a planner on top of it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/well-iam/humanoid-intern-challenge/blob/main/notebooks/pushing_world_model.ipynb)

![My clip, the model's prediction, the Panda replay](assets/gif-strip.gif)

*Left:* one of my test clips, which the model never saw in training (palm in green).\
*Middle:* what the model expected this moment to look like, 1 s before it happened, given my palm's moves. A prediction is a feature vector, not an image, so I show the training frame whose features are closest to it.\
*Right:* the Panda replaying my palm path in MuJoCo.\
Note: I picked a typical clip, not the best: the robot's box ends 4.5 cm from where my real box ended, and turned 41° differently.

## In short

✅ What worked:
- **Push-T** (a public dataset of a round pusher moving a T-shaped block): 2 s ahead, the predictor's error is 25% below the same model trained without the pusher's moves, and 59% below "nothing moves". My pass mark was 20%, fixed before any result. It passed on 1 of 3 training runs with different random starts.
- **My own 115 clips:** both pass marks I fixed in advance pass. The box's position is read from a real frame within 1.24 cm (2 cm needed). 1 s ahead, the predictor's error is 32% below the same model without my hand's moves (20% needed). My first 30 clips failed both; more clips fixed it.
- **The robot:** my palm paths drive a simulated Panda arm. After a push of about 15 cm, its box typically ends about 4 cm from where my real box ended. On all 35 near-misses, the arm passes the box without moving it.

❌ What didn't:
- **Near-misses:** when my hand passes the box without touching it, the predictor still moves the box 1.44 cm in 1 s. Reading the real, still box from the real frames already wobbles by 0.67 cm, so about half of that is real error.
- **Choosing a push:** the model ranks 32 candidate pushes roughly like the simulator does, but its single top pick is worse than a random one.
- **The box's angle** in the robot replay: 25° off.

## The idea

- **The eyes:** DINOv2-small, a ready-made image model, turns each frame into a 16 × 16 grid of feature vectors. I don't train it.
- **The predictor:** a small transformer (about 20M parameters), the only part I train. It takes the last 3 frames' features, where the pusher is, and how it will move next, and predicts the next frame's features. Feeding its guesses back in predicts 1, 2, 4 and 8 steps ahead.
- **The ruler:** a probe, a simple linear read-out, finds the object's position in real or predicted features. So every score is a distance on the table.

I compare the predictor with two rivals: **"nothing moves"**, and its **blind twin**, the same model trained the same way, which sees the pusher's position but not its next move. Then I give the predictor wrong moves (reversed or shuffled): if it really uses them, it should get worse. I only score moments where the object really moves; otherwise "nothing moves" wins for free.

Two pass marks, fixed before any result:

- **Gate 1, the ruler works:** on real frames, the probe finds the object within 4% of the table's width.
- **Validation gate, the predictor uses the moves:** 4 steps ahead, it is at least 20% better than both rivals, and wrong moves make it worse.

## Push-T results

Before recording anything, I tested the method on Push-T. It is a public dataset: a round pusher moves a T-shaped block on a flat board, and every position is recorded exactly. The board is 512 units wide. I trained on 100 recordings and kept 50 aside for testing. The model sees one frame every 0.5 s.

I also tried a second, much larger image encoder: V-JEPA 2. Everything else was the same.

**The question:** given the last frames and how the pusher will move, where will the T be 2 s (4 steps) later? The error is the distance between the predicted and the true position of the T, in board units. Lower is better. I count only moments where the T really moves in those 2 s.

Error 2 s ahead, on the 50 test recordings:

| | DINOv2-small | V-JEPA 2 |
|---|---:|---:|
| **the predictor** | **18.14** | **18.96** |
| the same model, blind to the pusher's moves | 24.08 | 22.80 |
| "nothing moves" | 44.55 | 44.44 |
| the predictor, given the pusher's moves reversed | 46.64 | 46.64 |
| the predictor, given the pusher's moves shuffled | 38.71 | 37.84 |

What this says, for DINOv2:

- The predictor is off by 18 units, about 3.5% of the board's width.
- It is 25% better than its blind twin, and 59% better than "nothing moves".
- Given the wrong moves, it gets much worse: reversed moves are as bad as "nothing moves". So it really uses the pusher's moves.

The plot shows the same comparison 1, 2, 4 and 8 steps ahead:

![Error vs horizon on Push-T, both encoders](assets/pusht-error-vs-horizon.png)

**The two pass marks, fixed before any result:**

| | DINOv2-small | V-JEPA 2 |
|---|---|---|
| **Gate 1:** the probe reads the T's position from real frames within 20 units (4% of the board) | ✅ 3.96 | ✅ 8.27 |
| **Validation gate:** the predictor is at least 20% better than its blind twin | ✅ 25% | ❌ 17% |

DINOv2 passed both, and it is the only encoder I used after this.

**Was the pass luck? Partly.** I trained the DINOv2 predictor and its blind twin twice more, with the same recipe and different random starts (seeds):

| seed | predictor | blind twin | needed: under 80% of the blind twin | result |
|---|---:|---:|---:|---|
| 0 (above) | 18.14 | 24.08 | 19.26 | ✅ pass |
| 1 | 17.95 | 21.31 | 17.05 | ❌ missed by 0.90 |
| 2 | 35.19 | 43.46 | 34.77 | ❌ training got stuck: both models barely beat "nothing moves" |

The predictor itself gives the same error on seeds 0 and 1 (18.14 and 17.95). What changed is the blind twin: on seed 1 it trained better, which raised the bar. So seed 0 passed partly because its blind twin happened to be weak. Per-seed plots: [assets/pusht-seeds.png](assets/pusht-seeds.png).

So the gate does not really separate the two encoders: V-JEPA 2 missed it by 0.72 units, less than the 0.90 that DINOv2's own seed 1 missed by. I kept DINOv2 for two reasons that do not depend on the seed: the probe reads the T's position from its features twice as precisely (3.96 vs 8.27 units), and its features take 2.7 times less disk space (3.5 vs 9.2 GB).

## Ranking test: can the model choose a good push?

A world model is useful if it can tell a good action from a bad one before trying it. So I asked it to choose.

**The test.** Push-T has a goal: a target place for the T. I took 50 moments from the test recordings. For each one, I made 32 different pusher paths, 4 s long: the path the human really took, plus 31 variations of it (turned, made longer or shorter, bent). Then:

1. The model imagined each path and predicted where the T would end. That gives its ranking of the 32 paths, from closest to the goal to furthest.
2. Push-T's own simulator played every path for real, which gives the true ranking.

![Ranking test](assets/ranking.png)

**What came out.** How far the T really ended from the goal, in board units, averaged over the 50 moments:

| path | distance to the goal |
|---|---:|
| the model's top pick | 79.8 |
| a random pick, on average | 73.8 |
| the path the human really took | 47.6 |
| the best of the 32 paths | 31.6 |

- ✅ **The ranking as a whole is better than chance.** Its rank correlation with the simulator's ranking (1 means the same order, 0 means unrelated) is +0.26 on average, and above 0 in 36 of the 50 moments. With the pusher's moves shuffled, it drops to 0.00, so the ranking comes from the moves.
- ❌ **But its favourite path is worse than a random pick** (79.8 against 73.8).

**Why the favourite is bad.** For its top picks, the model predicted the T would end 20.7 units from the goal; the simulator said 79.8. On those paths its prediction is 69 units off, against 42 on an average path. So the path the model likes most is usually the one where it is most wrongly optimistic: choosing the best-looking path chooses the model's biggest mistakes. This is a well-known trap when a robot plans with a learned model, and one reason I did not build a planner on top of it.

## My data

### Recording

I filmed my hand pushing a white box (17.8 × 5.8 × 3 cm) on a carpet, with a phone held 71 cm above the table. I recorded 9 takes. I clapped before each push, so the claps cut the takes into clips and I never had to touch the phone. That gave 119 clips. I dropped 4 where my hand came back into view to reset the box. That leaves **115 clips: 80 pushes, and 35 near-misses** where my hand passes the box without touching it.

### From video to numbers

- **The table:** a printed square marker (ArUco) lies on the table. It tells the software where the table is, so each frame is straightened into a top-down view of a 50 × 50 cm square, 8 frames per second.
- **The hand:** MediaPipe, a free hand-tracking tool, finds my hand. The pusher position is the centre of my palm.
- **The box:** the bright rectangle of the right size.
- **The end of a clip:** when my hand has been out of view for 0.75 s, so no clip includes my hand coming back.
- **Are the centimetres right?** I measured known objects in the straightened frames. The A4 sheet under the marker comes out within 0.28 cm of its real size, and the box's top within 0.77 cm.

### Training and testing

I kept **18 clips aside for testing**: 10 pushes and 8 near-misses. The models never see them in training. 8 of them are from my first 3 takes, chosen before I trained anything; the other 10 were drawn at random (with a fixed seed) from the later takes. The other **97 clips train** the models.

The model and the training recipe are the same as on Push-T, with one frame every 0.25 s. Nothing was tuned on the test clips.

### The first try failed; more clips fixed it

My first run used only my first 3 takes: 22 training clips and 8 test clips. It failed both pass marks. I recorded 6 more takes and reran everything unchanged. The same 8 test clips are still kept aside, so I can compare before and after on the same clips.

Before (22 training clips) and after (97 training clips), both graded on the same 8 test clips. Error 1 s (4 steps) ahead, in cm, lower is better. Only moments where the box really moves count:

| | before | after |
|---|---:|---:|
| **the predictor** | **3.78** | **3.15** |
| the same model, blind to my hand's moves | 4.46 | 4.61 |
| "nothing moves" | 5.83 | 6.34 |
| the predictor, given my hand's moves reversed | 5.29 | 6.45 |
| the predictor, given my hand's moves shuffled | 4.42 | 5.10 |

The two pass marks, the same as on Push-T:

| | before | after |
|---|---|---|
| **Gate 1:** the probe reads the box's position from real frames within 2 cm | ❌ 3.49 cm | ✅ 1.72 cm |
| **Validation gate:** the predictor is at least 20% better than its blind twin | ❌ 15% | ✅ 32% |

What this says:

- **Both gates now pass**, on the same test clips, with only the training data changed.
- **Wrong moves now hurt more.** Given my hand's moves reversed, the predictor is now as bad as "nothing moves" (6.45 against 6.34 cm); before, it was still better (5.29 against 5.83). So it relies on my hand's moves more.
- "Nothing moves" got worse (5.83 to 6.34 cm). It is read by the probe too. My guess: the first, weaker probe pulled every position toward the middle of the table, which hid part of the motion.

**The final model on all 18 test clips** (the 8 above, plus 10 from my later takes). The probe reads the box within 1.24 cm. 1 s ahead, the predictor is off by 2.82 cm, against 4.16 for its blind twin and 6.67 for "nothing moves": 32% and 58% better. It beats its blind twin on every one of the 10 test pushes. 2 s ahead the gap holds: 4.02 cm, against 5.96 and 11.84.

The plot shows the final model on all 18 test clips, 1, 2, 4 and 8 steps ahead:

![Error vs horizon on my clips](assets/clips-error-vs-horizon.png)

### Why more clips helped: the probe

With 22 training clips, the probe was 3.49 cm off even on real frames, so every score above had that error built in. Giving DINOv2 sharper frames (448 px instead of 224 px) did not help. More clips did:

![Probe error vs number of training clips](assets/probe-scaling.png)

To draw this, I trained the probe on more and more of the 97 training clips and tested it on the training clips it had not seen, 5 times over with different splits (5-fold cross-validation). The 18 test clips were never used. The probe gets under 2 cm somewhere between 24 and 48 clips. One caveat: clips from the same take can land on both sides of a split, which may make the curve look a little better than it is. The test-clip numbers (1.72 and 1.24 cm) agree with it.

### Near-miss test

When my hand passes the box without touching it, a good world model should predict that the box stays still.

![Near-miss test](assets/near-miss.png)

❌ On the 8 test near-misses, 1 s ahead, the predictor moves the box 1.44 cm. Reading the real, still box from the real frames already wobbles by 0.67 cm, so the model still imagines a small push that never happened. But it got better: the first run moved the box 2.37 cm. And it now moves the box less than its blind twin (2.17 cm), which the first run never did. So my hand's moves now tell it something about "the hand passed by".

### My palm paths drive the Panda

**The problem:** MediaPipe gives the centre of my palm, but the part of my hand that touches the box is a different point. \
**The fix:** at the moment of contact, I measure the gap from my palm to the nearest edge of the box, and shift the whole palm path by it.

The Panda's closed gripper then follows the shifted path, 1.5 cm above the table, in the MuJoCo simulator. The mink library is used for inverse kinematics. The simulated box starts where my real box started.

In 32 of the 80 pushes, MediaPipe found my hand too late: less than 0.25 s before the box moved, so I don't know where my palm was at contact. For those, I use the typical shift from the other 48 pushes, turned to the direction my hand moves.

![Panda replaying one of my pushes](assets/panda-push.gif)

A test push where my hand was found late, so the path uses the typical shift. It is a middle one, not the best: of the 7 such test pushes, 3 end closer and 3 further than this one (5.3 cm off).

How far the simulated box ends from where my real box ended. The replay involves no training, so I show all 115 clips:

| clips | typical error | worst error | how far my real box moved, typically |
|---|---:|---:|---:|
| 48 pushes, with their own shift | 3.8 cm | 11.1 cm | 14.4 cm |
| 32 pushes where my hand was found late | 4.7 cm | 19.1 cm | 16.4 cm |
| 35 near-misses | 0.0 cm: the simulated box never moved | 0.5 cm | 0.0 cm |

"Typical" is the middle value: half the clips do better, half worse. On my first 30 clips, the late pushes improved from 5.5 to 3.0 cm, because the typical shift now comes from 48 pushes instead of 12.

The near-misses stay near-misses on the robot. In [assets/panda-near-miss.gif](assets/panda-near-miss.gif), the gripper goes by the box without moving it. The same replay code also runs on Push-T paths: [assets/pusht-panda.gif](assets/pusht-panda.gif).

## Design choices

- **Prediction in feature space, scored in centimetres.** I never decode predictions to pixels. The linear probe turns any prediction into a position, so I can compare models in physical units against simple baselines.
- **Push-T first.** It has exact positions and its own simulator, so I could check the method before recording.
- **Pass marks fixed before results.** The gates, the training recipe and the ranking test setup were all committed before I ran them.
- **The pusher action is the observed move of the pusher, not the controller's command.** That is all hand tracking can give on my clips, so I used the same on Push-T.
- **DINOv2-small over V-JEPA 2 ViT-L.** The plan picked the encoder with the lower error 2 s ahead, which was DINOv2, but that gap is within seed luck. The reasons that hold up: the probe is twice as precise on its features, and they are 2.7 times smaller on disk (3.5 vs 9.2 GB).
- **The full 16 × 16 grid, no pooling.** On a 50 cm square, a 4 × 4 pooled cell would cover 12.5 cm, and the 5.8 cm wide box would be a small part of it.

Dropped, and why:

| option | why I dropped it |
|---|---|
| LIBERO | Its dependencies are old, and one would not build with current tools. Plain MuJoCo with a ready-made Panda model installed in a minute. |
| SmolVLA fine-tuning | 4 to 5 h on a big GPU, about 5 times longer on a free one. And my hand clips have no robot actions to learn from. |
| Planning with the model | 4 to 8 h more work, and the ranking test already showed that a planner would pick the model's mistakes. |
| Meta's released V-JEPA 2 robot predictor | Built for a bigger version of V-JEPA 2 than the one I used. |
| Predicting the object's angle | Out of time. The probe reads position only. |
| Decoding predictions to pixels | A second model to train (out of time). The nearest real frame is enough to show what a prediction means. |

## What didn't work

- **Near-misses:** the predictor still imagines a small push when my hand only passes by the box. Better than my first run, not solved. See [Near-miss test](#near-miss-test).
- **Choosing a push:** the model's single favourite path is worse than a random one. See [Ranking test](#ranking-test-can-the-model-choose-a-good-push).
- **The Push-T pass mark held on only 1 of 3 training runs.** On seed 2, training got stuck: both models ended up barely better than "nothing moves". I did not look into why. See [Push-T results](#push-t-results).
- **DINOv2 against V-JEPA 2 is not settled by the pass mark.** V-JEPA 2 missed it by less than DINOv2's own seed 1 did, and I trained V-JEPA 2 only once. I kept DINOv2 for its more precise probe and smaller features.
- **Hands found late:** in 32 of my 80 pushes, MediaPipe found my hand only just before the box moved, so I don't know where my palm was at contact. The robot replay has to guess the shift for those, and does worse (worst case 19.1 cm off). Of my 10 test pushes, only 3 have their own shift.
- **The box's angle in the robot replay** is typically 25° off. The Panda pushes with two closed fingers (3.8 × 2 cm); my hand pushes with a wide edge, so the box turns differently.
- **Box size on the new takes:** my box measured within 0.33 cm of its real size on the first 3 takes, but up to 0.77 cm off on all 9. I did not look into why.
- **Push-T paths on the Panda:** I also replayed Push-T's recorded paths on the Panda, followed blindly with no correction. Most end close (typically about 2 cm off at the Panda's scale), but the worst is far off. Push-T's own simulator has no friction with the floor; MuJoCo does. So the T drifts differently, and the human's later corrections then push it the wrong way.
- **One setup only:** one box, one carpet, one camera height, one person's hand. I did not test whether the model works on anything else.

## How to run

### Colab

Runtime > Run all. By default the notebook downloads the cached features, trained predictors and tracks from the Hugging Face dataset [`well-iam/humanoid-intern-challenge`](https://huggingface.co/datasets/well-iam/humanoid-intern-challenge), and rebuilds every figure and GIF on the CPU. It took about 12 min on a fresh Colab runtime (11 min on my machine with the GPU hidden). Set `RETRAIN = "clips"` to retrain the predictors on my clips (about 40 min on a T4, estimated) or `"all"` to also redo Push-T with DINOv2 (about 1.5 h, estimated). Push-T is downloaded from its original URL, never re-hosted.

An earlier test of the notebook ran on a T4 GPU runtime: Python 3.13.15, PyTorch 2.11.0+cu130.

### Locally

With [uv](https://docs.astral.sh/uv/), Python 3.12 and an NVIDIA GPU (I used a Quadro RTX 4000, 8 GB):

```bash
uv sync
export PWM_DATA=/path/to/pwm-data   # every stage reads and writes here
uv run pytest                       # fast tests, on a toy world
uv run pytest -m slow               # data, simulator and long-training tests
```

Then rerun every stage, from raw data to the README's figures:

```bash
bash scripts/reproduce.sh pusht   # Push-T: both encoders, the extra seeds, the ranking test, the Panda replay
bash scripts/reproduce.sh clips   # my clips
bash scripts/reproduce.sh all     # both, then copy the figures into assets/
```

Each model trains in about 15 min. The script lists every step in order, and each step is its own script in `scripts/`, so you can rerun one alone. The 1080p takes of my clips are not public: if `$PWM_DATA/clips/raw/` is empty, the script downloads the tracked clips from the Hugging Face dataset and starts from there. I wrote the script from the commands I ran, but did not run it end to end before the deadline.

To get all the cached results instead (9.6 GB):

```bash
uv run hf download well-iam/humanoid-intern-challenge --repo-type dataset --local-dir $PWM_DATA
```

## License

Code: MIT (see `LICENSE`). My clips and the cached results on Hugging Face: CC BY 4.0.
