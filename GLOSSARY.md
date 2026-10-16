# Pushing world model

An action-conditioned world model: given what the scene looks like and how the pusher will move, it predicts how the scene will change. It is checked first on a public reference dataset, then on my own phone recordings of my hand pushing an object on a table.

## Data

**Object**:
The thing being pushed: a white box in the clips, a T-shaped block in the reference dataset, a body of the same shape in the simulator.
_Avoid_: Box, wallet, block

**Pusher**:
Whatever pushes the object: my palm in the clips, the round agent in the reference dataset, the Panda's gripper in the simulator.
_Avoid_: Agent, end-effector, hand (when meaning the general role)

**Reference dataset**:
A public dataset with exact actions and object positions, used to check the method before my own data.
_Avoid_: Benchmark, pretraining data

**Episode**:
One pushing run in the shared format: a run from the reference dataset, or one of my clips once it is tracked.
_Avoid_: Trajectory, demo, rollout (when meaning recorded data)

**Take**:
One continuous recording that holds many clips, cut apart at claps.
_Avoid_: Batch, session, video

**Clip**:
One push or near-miss cut out of a take, from a fixed top-down phone camera.
_Avoid_: Video, sample

**Push clip**:
A clip in which the hand touches the object and pushes it toward the target.

**Near-miss clip**:
A clip in which the hand passes close to the object without touching it, so the object stays still.
_Avoid_: Negative clip, distractor

**Held-out clip**:
A clip or episode never used for training, kept only for testing.
_Avoid_: Validation clip, test sample

**Mirrored clip**:
A training clip flipped left to right, with its pusher actions flipped to match.
_Avoid_: Augmented clip

**Table marks**:
Four small marks at the corners of a measured rectangle on the table, which tie image pixels to centimetres on the table.

**Check mark**:
A fifth mark at a measured spot, used only to measure the error of the pixel-to-centimetre conversion.

## State and action

**Step**:
The fixed time between two consecutive frames the model sees.
_Avoid_: Tick, timestep, frame (when meaning time)

**Pusher position**:
Where the pusher is on the table.

**Palm position**:
The pusher position in the clips: where the centre of the palm is on the table, in centimetres.
_Avoid_: Hand pose, wrist position

**Object position**:
Where the centre of the object is on the table.
_Avoid_: Object state, box position

**Pusher action**:
How far the pusher position moves over one step, as a 2D displacement on the table.
_Avoid_: Command, control, hand action

**Pusher path**:
The sequence of pusher positions over a clip or episode.
_Avoid_: Trajectory, palm path, hand track

## Model

**Frame embedding**:
The frozen encoder's description of one frame.
_Avoid_: Latent, feature vector, state

**Predictor**:
The only trained model: it maps the recent frame embeddings, the pusher position and a pusher action to the next frame embedding.
_Avoid_: World model (when meaning just this part), dynamics net

**Rollout**:
Predicting several steps ahead by feeding the predictor its own predictions.
_Avoid_: Unroll, autoregressive generation

**Horizon**:
How many steps ahead a prediction reaches.

**Probe**:
A linear read-out that recovers the object position from a frame embedding, real or predicted.
_Avoid_: Decoder, head

## Evaluation

**Moving window**:
A stretch of a clip or episode in which the object really moves within the horizon being scored.
_Avoid_: Active segment, contact window

**Static baseline**:
The prediction that nothing changes: the last real frame embedding is repeated at every horizon.
_Avoid_: Copy baseline, identity baseline

**Action-blind predictor**:
A predictor with the same design and training but no pusher action as input.
_Avoid_: Unconditioned model

**Wrong-action test**:
Running a rollout with deliberately wrong pusher actions to check that the predictor really depends on them.
_Avoid_: Ablation, counterfactual test

## Simulation

**Replay**:
The simulated Panda's closed gripper following a pusher path, with the object starting where it really started.
_Avoid_: Retargeting (when meaning the whole run), playback

**Candidate path**:
A pusher path made by perturbing a recorded one, to be judged by the predictor and by a simulator.

**Ranking test**:
Ranking candidate paths by where the predictor expects the object to end up, then checking that ranking against a simulator.
_Avoid_: Planning
