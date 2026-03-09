# Joint-Angle 3D Palm-Down RL

Standalone subproject for training a policy that controls a MyoSuite human arm in **3D** using **basic joint-angle commands**.

## Simple Reach (Recommended Starting Point)
If you want a mechanical-arm-style **position-only reaching** baseline with minimal reward shaping, use:
- `simple_reach_env.py`
- `train_simple_reach.py`
- `render_simple_reach.py`

Train:
```bash
python -m joint_angle_3d_palm_rl.train_simple_reach \
  --total-steps 1000000 \
  --n-envs 4 \
  --tb-log-dir /absolute/path/to/tensorboard_logs \
  --randomize-start-pose \
  --start-pose-trials 64 \
  --reward-mode distance \
  --action-mode delta
```

Render:
```bash
python -m joint_angle_3d_palm_rl.render_simple_reach \
  --model /absolute/path/to/best_model.zip
```

It is designed for:
- random target setting in 3D space,
- random reset hand pose (position + attitude) with retry sampling,
- minimizing hand-target position error,
- soft palm-down preference, mainly enforced near target (no hard lock),
- steady target-hold success (default requires 10 consecutive in-threshold steps),
- low control cost and smooth/fluency-friendly motion.

This project references the original shoulder GUI ideas from the source development repository, especially palm posture handling and palm normal estimation.

## Remote Setup
Repository-relative working directory:

- repo root: `joint-angle-3d-palm-rl`
- this subproject: `joint-angle-3d-palm-rl/joint_angle_3d_palm_rl`

To continue this work on another machine:

```bash
git clone https://github.com/NUS-AI4HCI/joint-angle-3d-palm-rl.git
cd joint-angle-3d-palm-rl
```

Interactive render helper:

```bash
./run_simple_drag_render.sh
```

Shared-autonomy handoff document for the next agent:

```bash
joint_angle_3d_palm_rl/WEAK_COORD_SHARED_AUTONOMY_HANDOFF.md
```

## Folder contents
- `env.py`: `JointAnglePalmDownReach3DEnv` (continuous joint-angle RL env)
- `train.py`: PPO training entrypoint
- `drag_policy_demo.py`: interactive target dragging (XY drag + Z slider + embedded 3D render panel)
- `requirements.txt`

## Action space
- Continuous `Box(-1, 1, n_joints + 1)` by default.
- Each action dimension maps to a delta joint angle:
  - `delta_q = action * action_max_delta_deg`.
- Last action dim controls bounded root yaw:
  - `absolute` mode (default): action maps directly to a yaw angle in range.
  - `delta` mode: action increments yaw each step.
- Joint profile defaults to `opensim_arm_wrist`:
  - `elv_angle, shoulder_elv, shoulder1_r2, shoulder_rot, elbow_flexion, pro_sup, deviation, flexion`

## Reward
Per step:
- `- distance_weight * ||target - hand||` (main objective)
- `+ progress_weight * (prev_dist - dist)`
- `- directness_weight * directness_error` (encourages direct motion toward target)
- `- away_penalty_weight * max(-progress, 0)` (extra penalty when moving away)
- `- palm_weight * palm_down_error`
- `- stability_weight * stability_penalty` (reduces jitter near the goal)
- `- action_cost_weight * mean(|action|)` (control cost)
- `- smoothness_weight * mean((action - prev_action)^2)` (fluency/smoothness)
- `- time_penalty`
- `+ success_bonus` when error < `success_radius` (one-time per episode/target)

## Train
From repository root:

```bash
python -m joint_angle_3d_palm_rl.train \
  --total-steps 500000 \
  --n-envs 1 \
  --tb-log-dir /absolute/path/to/tensorboard_logs \
  --joint-profile opensim_arm_wrist
```

TensorBoard:
```bash
tensorboard --logdir /absolute/path/to/tensorboard_logs --port 6006
```

If missing package in your env:
```bash
conda run -n vibe pip install tensorboard
```

Training defaults now include:
- learning-rate annealing (linear schedule to 0 over training),
- periodic evaluation by distance/success metrics (not raw reward),
- automatic best model saving as `best_model.zip`,
- non-trivial reset targets (`target_min_start_dist`, `target_max_start_dist`) to reduce early fake reward spikes,
- root yaw enabled with bounded action output,
- soft-only palm handling by default (`--palm-lock-alpha 0`),
- easier initial curriculum (`--retarget-interval-steps 0`, `--terminate-on-success` default on).

Optional custom Myo XML:

```bash
python -m joint_angle_3d_palm_rl.train \
  --model-path /absolute/path/to/custom_model.xml
```

## Interactive drag demo
After training:

```bash
python -m joint_angle_3d_palm_rl.drag_policy_demo \
  --model /absolute/path/to/model_final.zip \
  --vecnormalize /absolute/path/to/vecnormalize.pkl
```

Controls:
- Left mouse drag on XY panel: move target in XY.
- Z slider: move target in Z.
- Embedded panel: live 3D model render in the same UI window.
- Embedded 3D panel camera: left-drag rotate, right-drag pan, middle-drag lift, mouse-wheel zoom, `R` reset camera.
- Separate interactive MyoSuite viewer is enabled by default (same `env.render()` style as shoulder GUI).
- `Q`: quit.

Drag behavior note:
- Manual drag targets are now applied directly (no reachable-point projection), so cursor motion is continuous and does not jump between scattered points.
- Demo defaults now disable in-episode random retargeting, and also disable auto-retarget on success, to avoid random target switches during manual control.
- Manual target updates now use a tip-referenced target-line state (`line_from_tip`) so policy inference is consistent with training-time reach setup.

Notes:
- `model_final.zip` is always saved.
- Best checkpoint is saved as `best_model.zip`.
- If `vecnormalize.pkl` exists, pass it with `--vecnormalize`.
- You can enlarge XY workspace panel with:
  - `--workspace-samples`, `--workspace-margin`, `--min-xy-width`
  - or explicit bounds: `--x-min --x-max --y-min --y-max`.
