---
license: cc-by-4.0
pretty_name: Pushing world model, my hand pushing a box
tags:
- robotics
- world-model
- video
- manipulation
size_categories:
- n<1K
---

# Pushing world model: my clips and cached results

Data and cached results for the [pushing-world-model project](https://github.com/well-iam/humanoid-intern-challenge).
A small world model predicts, in DINOv2 feature space, where an object goes when a pusher moves. It is trained on
Push-T, then on 115 clips of my own hand pushing a box on a table. My recorded palm paths then drive a simulated
Franka Panda in MuJoCo.

The project's Colab notebook downloads this dataset and rebuilds every figure and GIF from it, without a GPU.

```python
from huggingface_hub import snapshot_download

snapshot_download("<HF_REPO_ID>", repo_type="dataset", local_dir="pwm-data")
```

## What is in it

**My clips** (`clips/`), recorded on 16 October 2026 with a phone 71 cm above a dark carpet, pushing a white box
(17.8 x 5.8 x 3 cm) inside a 50 x 50 cm square. 9 takes, 119 clips cut at claps; 4 junk clips left out (the hand
resetting the box; the list and reasons are in the tracking report), so 115 clips: 80 pushes, 35 near-misses.

| path | what |
|---|---|
| `clips/videos/<take>.mp4` | the 9 takes, 720p H.264 with audio (the claps cut the takes into clips) |
| `clips/episodes/<clip>.npz` | one episode per clip: the straightened 512 x 512 px top-down frames at 8 fps, palm and box tracks (cm), what was seen, the box angle, the hand direction, the split and the kind (push or near-miss) |
| `clips/tracks.csv` | the same tracks as one table, one row per frame |
| `clips/split.json` | the held-out split: 18 clips (10 pushes, 8 near-misses): the first 3 takes' 8, plus 4 pushes and 6 near-misses of the later takes drawn with a fixed seed |
| `clips/tracking/` | the tracking report and one overlay GIF per take |
| `clips/dinov2/embeddings/<clip>.npy` | DINOv2-small patch embeddings, fp16, (frames, 256, 384) |
| `clips/dinov2/predictor.pt`, `predictor-blind.pt` | the trained predictor and the action-blind predictor |
| `clips/dinov2/` | results, reports and figures: error vs horizon, near-miss test, retrievals, training curves, probe error vs number of training clips (`probe-scaling.png`) |
| `clips/dinov2/only-takes/` | the same model scored on the first 3 takes' 8 held-out clips only (for the before/after) |
| `clips/panda/` | the Panda replay of every clip (`replay.json`) and GIFs |
| `clips/dinov2-30clips/`, `clips/panda-30clips/` | the first run, on the first 30 clips (3 takes): results, figures, predictors, replay |
| `clips/gif-strip/` | a held-out clip, the predictor's view one second ahead (nearest real frame) and the Panda replay, side by side |
| `clips/logs/` | training logs (batch loss every 100 steps) |
| `clips/aruco-a4.pdf` | the printed marker that fixes the table frame |

Table frame: cm, origin at the square's top-left corner as seen from the camera, x to the right, y down.
Palm = mean of MediaPipe hand landmarks 0, 5, 9, 13, 17, height-corrected.

**Push-T results** (`pusht/`): DINOv2-small and V-JEPA 2 ViT-L embeddings of the 150 Push-T episodes used
(fp16, `pusht/<encoder>/embeddings/pusht-<index>.npy`, index = position in Push-T's file), the trained
predictors (three seeds for DINOv2), results, the ranking test and the Panda replay. Push-T itself is **not**
included: the code downloads it from its original URL
(`https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`, from Chi et al., *Diffusion Policy*, 2023).

Code, the exact commands and the results are in the project's repository and README.

## License

CC BY 4.0. The Push-T embeddings are computed from the Push-T dataset; its own terms apply to the underlying data.
