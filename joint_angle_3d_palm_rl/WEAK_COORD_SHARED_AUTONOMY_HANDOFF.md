# Weak-Coordinate Shared Autonomy Handoff

This document is for another coding agent that must continue the project without the original chat session.

## Copy-Paste Brief For Another Agent

You are working at the repository root:

- `joint-angle-3d-palm-rl`

Relevant existing subproject:

- `joint_angle_3d_palm_rl`

Default branch to use:

- `main`

Remote pull / checkout:

```bash
git clone https://github.com/NUS-AI4HCI/joint-angle-3d-palm-rl.git
cd joint-angle-3d-palm-rl
```

Existing context:

- This repo already contains a MyoSuite-based low-level human-arm RL reaching project.
- The current low-level policy is good for smooth local reaching in simulation.
- The real deployment target is not a robot arm. It is a human user wearing a head-mounted Intel RealSense D435i camera and a force-feedback finger ring.
- The user does not want full body / torso / shoulder coordinate estimation as the core dependency.
- The user wants a weak-coordinate shared-autonomy stack:
  - no rigid global body/world frame requirement,
  - no first-principles full human kinematic reconstruction,
  - top layer gives haptic guidance,
  - user performs the final contact / grasp,
  - system strongly warns or overrides guidance only near collision.

What to build:

- A new implementation path for weak-coordinate shared autonomy.
- The top layer should use image/depth/IMU information, not a stable torso frame.
- The top layer should output haptic guidance commands, not joint angles.
- Use the existing low-level RL policy only as a simulation surrogate / local motion prior, not as the real-world direct human controller.

Do not do these things:

- Do not make stable torso/chest/shoulder coordinates the core dependency.
- Do not assume a fixed robot-arm base.
- Do not build an end-to-end direct joint-control policy for real deployment.
- Do not make full-body SLAM or human pose reconstruction the first milestone.
- Do not require a precise persistent 3D world frame for the first version.

The correct first-version framing is:

- camera-centric, frame-by-frame, local, weak-coordinate shared autonomy.
- Each update only answers:
  - left or right correction,
  - up or down correction,
  - safe to advance or stop.

## Existing Files To Read First

- `joint_angle_3d_palm_rl/README.md`
- `joint_angle_3d_palm_rl/simple_reach_env.py`
- `joint_angle_3d_palm_rl/train_simple_reach.py`
- `joint_angle_3d_palm_rl/render_simple_reach.py`

Why:

- `simple_reach_env.py` is the current cleanest low-level local reaching environment.
- `train_simple_reach.py` shows how the current local policy is trained.
- `render_simple_reach.py` shows how the current policy is visualized and interactively exercised.

## Problem Restatement

The user has:

- head-mounted `D435i`,
- Jetson Orin Nano,
- a force-feedback finger ring,
- a human arm, not a robot arm,
- a moving head and moving body,
- no desire to solve full-body coordinate estimation first.

The target application is assistive obstacle-avoiding reach / grasp for visually impaired users.

The right abstraction is not:

- "recover a stable body/world coordinate frame, then plan grasping in 3D."

The right abstraction is:

- "at each short time step, use the current camera view, depth, and gravity to produce a safe local guidance command for the user."

This is shared autonomy:

- user intent remains primary,
- autonomy only biases motion toward safe approach,
- autonomy strongly warns / blocks only when risk is high,
- user performs the final touch and grasp.

## Core Design Decision

Use a weak-coordinate representation.

This means:

- use instantaneous image-plane geometry,
- use depth differences,
- use gravity from IMU,
- optionally use a hand marker only as a moving point anchor,
- do not depend on a persistent torso/chest/shoulder world model.

The weak-coordinate top layer should operate on:

- target image position and depth,
- hand anchor image position and depth,
- local obstacle clearances from the depth map,
- gravity direction in camera coordinates,
- confidence flags.

## Recommended New Module Structure

Suggested new folder:

- `joint_angle_3d_palm_rl/weak_shared_autonomy`

Suggested files:

- `__init__.py`
- `weak_state.py`
- `tag_tracking.py`
- `depth_safety_field.py`
- `collision_geometry.py`
- `shared_policy.py`
- `haptic_mapping.py`
- `state_machine.py`
- `sim_demo.py`
- `ros_bridge.py`

Suggested purpose of each file:

- `weak_state.py`
  - define the weak-coordinate state container.
- `tag_tracking.py`
  - detect / parse target anchor and hand anchor.
- `depth_safety_field.py`
  - compute directional clearances from depth.
- `collision_geometry.py`
  - define hand / forearm approximate geometry and collision sampling.
- `shared_policy.py`
  - turn weak state into a local assistive command.
- `haptic_mapping.py`
  - map assistive command into three finger-force outputs.
- `state_machine.py`
  - search / align / advance / avoid / dock / stop transitions.
- `sim_demo.py`
  - run the shared-autonomy logic in simulation first.
- `ros_bridge.py`
  - expose inputs / outputs for deployment integration.

## Weak State Definition

Use a lightweight state vector instead of explicit rigid-body coordinates.

Suggested state:

```text
s = {
  u_h, v_h, z_h,          # hand anchor in image/depth
  u_t, v_t, z_t,          # target anchor in image/depth
  du, dv, dz,             # target minus hand
  gravity_cam,            # gravity direction from IMU in camera frame
  clr_left,
  clr_right,
  clr_up,
  clr_down,
  clr_forward,
  hand_clearance,
  forearm_clearance,
  target_conf,
  hand_conf,
  target_visible,
  hand_visible,
  mode,
}
```

Notes:

- `u` and `v` are normalized image coordinates.
- `z` is robust depth, ideally median depth inside a small patch around the anchor.
- `du`, `dv`, `dz` are the only target-relative quantities needed for first version guidance.
- `gravity_cam` stabilizes "up/down" without needing torso estimation.

## Anchors

First version should use fiducials.

Target anchor:

- put an `AprilTag` on the target object.
- output `(u_t, v_t, z_t)` and confidence.

Hand anchor:

- put an `AprilTag` or a small rigid marker board on the back of the hand or wrist.
- use it only as a moving anchor point.
- do not treat it as the definition of a full stable hand frame for the whole system.

Why:

- hand tag is useful as a point / patch tracker,
- but it is too fragile to serve as the global coordinate backbone,
- especially near final grasp when occlusion and rotation increase.

## Depth Safety Field

Implement directional clearances directly from the depth map.

Minimum viable behavior:

- remove invalid depth,
- optionally remove the floor plane using gravity,
- remove target-tag pixels,
- remove hand-tag pixels,
- evaluate free space around the hand-to-target corridor.

Compute:

- `clr_left`
- `clr_right`
- `clr_up`
- `clr_down`
- `clr_forward`

One practical method:

- define a 3D corridor between hand anchor and target anchor,
- dilate it by a safety radius,
- sample nearby depth points,
- compute minimum clearance for several candidate directional nudges.

Candidate nudges:

- left
- right
- up
- down
- forward

Each candidate gets a cost:

```text
J = w_goal * future_goal_error
  + w_obs * 1 / (future_clearance + eps)
  + w_smooth * command_change
  + w_vis * target_visibility_penalty
```

## Collision Geometry

Yes, weak-coordinate shared autonomy still needs collision geometry.

But do not start with a full human mesh.

First version geometry:

- one hand box or capsule,
- one forearm capsule,
- optional upper-arm capsule later.

Recommended first approximation:

- hand:
  - length `0.10-0.12 m`
  - width `0.06-0.08 m`
  - thickness `0.02-0.04 m`
- forearm:
  - length `0.20-0.25 m`
  - radius `0.03-0.04 m`

How to place them:

- hand box is centered on the hand anchor.
- forearm capsule is placed behind the hand anchor using either:
  - recent motion direction, or
  - a filtered hand marker orientation if available.

Do not block implementation on exact elbow estimation.

Collision check:

- sample points on hand box and forearm capsule,
- query their nearest depth-based obstacle distance,
- define:
  - `hand_clearance`
  - `forearm_clearance`
- use the minimum for hard stop behavior.

## Shared-Autonomy State Machine

Implement an explicit state machine.

Recommended states:

- `search`
- `align`
- `advance`
- `avoid`
- `dock`
- `stop`

Transitions:

- `search`
  - target not visible or low confidence.
- `align`
  - target visible, but `|du|` or `|dv|` still large.
- `advance`
  - target visible, roughly aligned, forward clearance acceptable.
- `avoid`
  - forward direction risky, but a lateral / vertical detour exists.
- `dock`
  - target close enough for final human-led approach.
- `stop`
  - hard safety threshold violated.

Initial threshold suggestions:

- align gate:
  - `|du| < 0.08`
  - `|dv| < 0.08`
- dock depth:
  - `dz < 0.10-0.15 m`
- warn clearance:
  - `0.06 m`
- hard stop clearance:
  - `0.03 m`

These are seed values only.

## Haptic Output Design

Do not encode raw global `x/y/z`.

Encode local correction and risk.

Recommended three-channel mapping:

- index finger:
  - horizontal correction
- middle finger:
  - vertical correction
- ring finger:
  - advance permission / danger / stop

Interpretation:

- user should try to reduce pressure,
- not decode an absolute coordinate value.

If each finger can only express one scalar force:

- use amplitude for magnitude,
- use pulse pattern for sign / type,
- reserve a special unmistakable pattern for stop.

Suggested pattern scheme:

- steady pressure:
  - one direction
- double pulse:
  - opposite direction
- sparse pulse:
  - safe to advance slowly
- continuous strong pressure:
  - stop
- strong repeating pressure:
  - back off / retreat

If the hardware supports directional force on finger sides:

- use actual directionality instead of pulse-sign coding.

## Shared Policy Output

The high-level policy should not output joint angles.

Output:

```text
a = {
  cmd_x,      # left/right corrective guidance
  cmd_y,      # up/down corrective guidance
  cmd_fwd,    # forward permission / caution
  risk,       # scalar risk
  mode,       # state machine mode
}
```

This action is then mapped into haptic outputs:

```text
h = {
  f_index,
  f_middle,
  f_ring,
  pattern_index,
  pattern_middle,
  pattern_ring,
}
```

## Use Of Existing Low-Level RL Model

Important:

- The current low-level RL policy should remain a simulation asset.
- It is not the real-world actuator for the human user.

How it should be used:

- simulation surrogate for local reaching motion,
- synthetic data generation,
- evaluation of how a human-like arm may respond to subgoal guidance.

Do not design the real shared-autonomy deployment assuming the low-level policy physically drives the user arm.

Instead:

- in simulation:
  - low-level RL can track subgoals.
- in reality:
  - top layer drives haptic guidance only.

## Training Strategy

Do not start with top-level RL.

Recommended sequence:

1. Rule-based baseline

- implement the state machine and hand-crafted safety field first.
- verify the haptic language.

2. Collect imitation data

Log tuples:

- weak state `s`,
- rule-based command `a`,
- resulting hand motion,
- target outcome,
- collisions / near misses.

3. Train a small learned policy

Input:

- weak state only, or
- weak state + low-resolution depth crop.

Output:

- the same shared policy command `a`.

4. Optional later refinement

- residual RL over the rule-based policy,
- but only after the behavior and safety language are already stable.

## Sim-First Demo

Build a simulation-first demo before touching the real ring.

Suggested script:

- `sim_demo.py`

Responsibilities:

- spawn or reuse the existing simple reach environment,
- compute weak state from rendered / synthetic anchors,
- run the shared-autonomy state machine,
- visualize:
  - target,
  - hand anchor,
  - corridor,
  - candidate nudges,
  - clearances,
  - resulting haptic outputs.

The purpose is not photorealism.

The purpose is to debug:

- haptic semantics,
- safety transitions,
- collision geometry,
- thresholding.

## Suggested ROS / Runtime Interfaces

Even if ROS2 is not fully implemented immediately, keep the interfaces explicit.

Suggested topics:

- `/camera/color/image_raw`
- `/camera/depth/image_rect_raw`
- `/camera/imu`
- `/assist/target_anchor`
- `/assist/hand_anchor`
- `/assist/weak_state`
- `/assist/shared_command`
- `/assist/haptic_command`
- `/assist/debug_clearance`

Suggested message contents:

- `target_anchor`
  - image center
  - depth
  - confidence
- `hand_anchor`
  - image center
  - depth
  - confidence
- `shared_command`
  - `cmd_x`
  - `cmd_y`
  - `cmd_fwd`
  - `risk`
  - `mode`
- `haptic_command`
  - per-finger amplitude
  - per-finger pattern id
  - timestamp

## First Milestone

The first milestone is not "real-world blind grasping."

The first milestone is:

- target tag visible,
- hand tag visible,
- depth safety corridor working,
- three-finger haptic command computed,
- state machine transitions working,
- stop behavior triggered correctly,
- simulation demo running.

Acceptance criteria:

- if target moves left/right/up/down, the system changes guidance correctly,
- if a depth obstacle blocks straight reach, the system switches to avoid,
- if clearance collapses, the system switches to stop,
- near target, the system enters dock and reduces aggressive guidance,
- hand / forearm collision proxies affect the warning logic.

## Explicit Non-Goals For First Version

- full body coordinate estimation,
- stable torso frame estimation,
- perfect 6D hand pose,
- full articulated human collision mesh,
- object-agnostic grasp synthesis,
- end-to-end RL from RGB-D to human joint control.

## Practical Build Order

1. Create `weak_shared_autonomy` package/folder.
2. Implement `weak_state.py`.
3. Implement `tag_tracking.py` using simple fiducial anchors.
4. Implement `depth_safety_field.py`.
5. Implement `collision_geometry.py` with hand + forearm only.
6. Implement `state_machine.py`.
7. Implement `haptic_mapping.py`.
8. Implement `shared_policy.py` by combining state machine + direction costs.
9. Implement `sim_demo.py`.
10. Add logging and plotting.
11. Only then consider learned top-layer replacement.

## Final Guidance To The Next Agent

The implementation should optimize for:

- robustness,
- inspectability,
- debuggability,
- low-latency local guidance,
- hard safety overrides.

Do not optimize first for:

- elegance of full 3D coordinate formulation,
- perfect human geometry,
- end-to-end learning.

The first version should be understandable enough that the user can watch the signals, inspect the guidance decisions, and trust the stop behavior.
