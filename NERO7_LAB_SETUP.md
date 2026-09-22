# Nero7 Lab Setup

This documents the Nero7 (AgileX NERO 7-DOF arm + Piper gripper) integration on
branch `nero_sim`, and how to run it.

## What this is

- **Robot**: `Nero7` (`robosuite/models/robots/manipulators/nero_robot.py`), a 7-DOF
  AgileX NERO arm with an AgileX Piper gripper, using `OSC_POSE` control
  (`robosuite/controllers/config/robots/default_nero7.json`).
- **Environment**: `LiftLabSetup1` (`robosuite/environments/manipulation/lift_labsetup1.py`)
  reproduces a real lab table: 0.76m x 1.34m, wood-textured (a real photo of the
  table, color-corrected), with Nero7's base at the world origin, mounted 8cm from
  the table's near edge and 73cm from its left edge. A dark grey 4cm cube spawns in
  a verified-reachable region of the table for pick tasks.
- **`StackLabSetup1`** (`robosuite/environments/manipulation/stack_labsetup1.py`):
  the same table/robot/camera setup, but with two matching 4cm dark-grey cubes
  (cubeA/cubeB) for a pick-and-stack task instead of Lift's single cube.
- **Camera**: `"d435"`, a fixed camera modeled after an Intel RealSense D435's RGB
  FOV (42.5deg vertical), attached to the robot base in a world-aligned frame (the real base frame; the sim
  `base_link` is yawed 180deg) at the real camera's
  hand-eye-calibrated pose (`T_BASE_D435` in
  `robosuite/models/arenas/lab_wood_table_arena.py`, OpenCV optical convention).

## Commands

Run everything from the repo root. All viewing/picking lives in one script,
`robosuite/scripts/pick.py`.

**Pick the cube** (scripted approach, face-align to the cube's nearest face, grasp,
and lift -- 100% success in a 20-trial batch). Note: an earlier version rotated an
extra 90deg off the nearest-face axis so the gripper's narrow profile (not its flat
body) faced the d435 camera, keeping the cube visible during the grasp; that offset
was pushing joint 6 into its actual angle limit (confirmed by checking qpos against
jnt_range), so it's been removed -- the gripper's flat body may occlude the cube
from d435 during the grasp now, in exchange for full reliability):
```bash
python robosuite/scripts/pick.py                    # on-screen viewer, default camera
python robosuite/scripts/pick.py --headless          # no display; prints success/diagnostics only
python robosuite/scripts/pick.py --camera d435       # on-screen viewer, watch from the d435 camera
```

**Just look around** (no pick):
```bash
python robosuite/scripts/pick.py --view --camera d435                        # live viewer
python robosuite/scripts/pick.py --view --camera agentview                   # any other camera
python robosuite/scripts/pick.py --view --camera d435 --save frame.png       # headless, save one frame
python robosuite/scripts/pick.py --view --save frame.png --width 1920 --height 1080
```

**Load it yourself** (e.g. for a policy or further scripting):
```python
import robosuite
env = robosuite.make(
    env_name="LiftLabSetup1",
    robots="Nero7",           # default, can be omitted
    has_renderer=True,        # or False + has_offscreen_renderer=True for headless
    has_offscreen_renderer=False,
)
env.reset()
```

**Collect a demonstration dataset** (scripted pick, run many times, saved to the
standard robosuite hdf5 format via `robosuite/scripts/collect_lift_data.py`; the
policy itself lives in `robosuite/scripts/scripted_lift_policy.py`):
```bash
python robosuite/scripts/collect_lift_data.py                                  # 20 episodes, headless, -> nero_datasets/lift/demo.hdf5
python robosuite/scripts/collect_lift_data.py --num-episodes 50                # more episodes
python robosuite/scripts/collect_lift_data.py --directory my_datasets/lift     # different output location
python robosuite/scripts/collect_lift_data.py --render                        # watch it collect, on-screen
```
Only successful episodes are kept. To try just the policy once without collecting:
```bash
python robosuite/scripts/scripted_lift_policy.py            # on-screen viewer
python robosuite/scripts/scripted_lift_policy.py --headless # no display; prints success only
```
To play a collected dataset back (needs a display):
```bash
python robosuite/scripts/playback_demonstrations_from_hdf5.py --folder nero_datasets/lift
python robosuite/scripts/playback_demonstrations_from_hdf5.py --folder nero_datasets/lift --use-actions
```

**Stack cubeA on cubeB** (ported from the HumanEgo branch's `scripted_stack_policy.py`
essentially unchanged -- tolerance/hold waypoints, yaw folded relative to the arm's
current orientation, and roll/pitch actively leveled throughout via
`scripted_wipe_policy.py`'s `get_orientation_action`). Needs `--max-phase-steps 300`
on this robot (the file's own default of 150 isn't enough, same as every other
script here). Currently 5/8 (62.5%) success -- grasping is reliable, but placing
cubeA precisely on cubeB is the main failure mode (see `session_summary.txt` item 32
for the diagnosis and what to try next):
```bash
python robosuite/scripts/scripted_stack_policy.py --max-phase-steps 300            # headless (default)
python robosuite/scripts/scripted_stack_policy.py --max-phase-steps 300 --render   # on-screen viewer
```

**Run the controller sanity tests:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_controllers/test_composite_controllers.py -k Nero7 -v
```
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` works around a ROS `launch_testing` pytest
plugin on this machine that otherwise crashes collection -- unrelated to robosuite.)

## Known limitations

- The wrist (joints 6/7) is torque-limited, and joint 6 specifically has a narrow
  angle range (-44 to +54deg) that some grasp orientations push right up against --
  both cause the gripper to tilt off-vertical during large or fast motions, most
  noticeably lifting. These are real hardware constraints (confirmed directly by
  checking qpos against joint ranges), not controller bugs. Cube placement in
  `LiftLabSetup1` is restricted to a region where this stays modest.
- Full details of everything built, fixed, and verified this session (bugs found,
  gain tuning, mesh fixes, reachability scan, etc.) are in `session_summary.txt`.
