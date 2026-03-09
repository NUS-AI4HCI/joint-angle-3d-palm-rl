# joint-angle-3d-palm-rl

Minimal repository export for the MyoSuite human-arm reaching project.

This repository intentionally includes only the files needed to:

- train the local reaching policy,
- render and interact with the simple reach demo,
- continue the weak-coordinate shared-autonomy implementation,
- avoid cloning unrelated large files from the original source repository.

## Contents

- `joint_angle_3d_palm_rl/`
- `run_simple_drag_render.sh`

## Quick Start

Clone:

```bash
git clone https://github.com/NUS-AI4HCI/joint-angle-3d-palm-rl.git
cd joint-angle-3d-palm-rl
```

Install Python dependencies:

```bash
pip install -r joint_angle_3d_palm_rl/requirements.txt
```

Render the latest simple-reach model:

```bash
./run_simple_drag_render.sh
```

Main documentation:

- `joint_angle_3d_palm_rl/README.md`
- `joint_angle_3d_palm_rl/WEAK_COORD_SHARED_AUTONOMY_HANDOFF.md`

