# G1_amo_play

Single-terminal LeRobot-v2.1 dataset replay on the Unitree G1 humanoid,
using the AMO low-level controller for legs+waist and the recorded
upper-body / Dex3-1 hand trajectories from the dataset.

This is a portable, vendor-everything-needed cut of the
[Psi0](https://github.com/physical-superintelligence-lab/Psi0) deploy
pipeline. It collapses the original two-process VLA architecture
(`Psi0/.venv` inference server + `psi_deploy` conda client over Unix
socket) into one Python process driven by parquet frames.

## What it does

```
parquet (action 36-d)                      G1 (29 body motors + 14 Dex3)
        │                                              ▲
        ▼  30 Hz feeder thread                         │ DDS rt/lowcmd + rt/dex3/*/cmd
+----------------------+                               │
| pred_action_buffer   |                               │
+----------------------+                               │
        │                                              │
        ▼  60 Hz control loop                          │
+----------------------------------------+             │
| AMO (15-d legs+waist) + arm[14] + hand[14] override  │
| RTC startup smoothing (50 ticks)       |             │
| pin.rnea -> tau_ff                     |─────────────┘
+----------------------------------------+
```

`action[28:36]` (8 dims: `rpy + height + vx,vy,vyaw + target_yaw`) feeds
AMO's lower-body command channel.
`action[14:28]` (14 dims) overrides arm PD targets.
`action[0:14]` (14 dims) overrides Dex3 hand cmd.

## Hardware prerequisites

- Unitree G1 with Dex3-1 hands, dev mode entered (`L2+B` then `L2+R2`)
- Workstation with NVIDIA GPU + CUDA 12.8 (~7 GB of torch+CUDA wheels;
  CPU-only also works if you swap the torch install line)
- Workstation network interface configured `192.168.123.X/24` directly
  to G1 (default `192.168.123.22`; override via `PSI_DDS_IFACE` env var)
- Ubuntu 22.04 (libstdc++ ≥ 6 from system or conda-shipped works)
- ~10 GB free disk for the conda env + this repo

## Repo layout

```
G1_amo_play/
├── play.py              # main script (single-process replay)
├── play.sh              # one-line wrapper (auto conda + LD_LIBRARY_PATH + DDS)
├── environment.yml      # conda env spec
├── README.md            # this file
├── LICENSE              # Apache 2.0 (inherited from upstream Psi0 / Unitree)
├── .gitignore
├── examples/
│   └── Pull_the_tray_episode_000000.parquet   # sample for smoke-test
├── teleop/              # vendored from Psi0/real/teleop, contains:
│   ├── master_whole_body.py   # the tested control loop wrapper
│   ├── adapter_jit.pt + adapter_norm_stats.pt + amo_jit.pt   # AMO assets
│   ├── robot_control/          # body / hand / IK / dex_retargeting (in-tree)
│   ├── utils/                  # logger / writers / weighted_moving_filter
│   ├── waist_logger.py / merger.py / constants.py / ...
│   └── (no VR / no image_server / no webrtc / no .bak)
├── assets/g1/           # G1 URDFs + STL meshes (~52 MB)
└── unitree_sdk2_python/ # vendored Unitree SDK (Psi0's modified fork)
```

## Quick start (cross-host)

### 1. Clone

```bash
git clone https://github.com/Ethan-Li-k/G1_amo_play.git
cd G1_amo_play
```

### 2. Build the conda env

Conda is required (the env ships its own libstdc++ ≥ 13 needed by
casadi 3.7). On a fresh box:

```bash
# install miniconda if you don't have it
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

# create the env (10-15 min)
conda env create -f environment.yml
conda activate g1_amo_play

# torch CUDA wheel (cu128 for RTX 40-series; switch URL for older CUDA)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# vendored Unitree SDK (must be -e because IDL files import via package path)
pip install -e ./unitree_sdk2_python
```

### 3. Smoke-test imports (no robot)

```bash
python -c "
import os, sys
sys.path.insert(0, 'teleop')
os.chdir('teleop')
import casadi, mujoco, pinocchio, pink, dex_retargeting, pyarrow, cyclonedds
from teleop.master_whole_body import RobotTaskmaster
print('all imports OK')
"
```

If this prints `all imports OK` you're ready for the robot.

### 4. Hang G1 in the rig and run

```bash
# default: bundled sample (Pull_the_tray episode 0)
bash play.sh

# any other parquet
bash play.sh /path/to/your/episode_000007.parquet

# different network interface (default 192.168.123.22)
PSI_DDS_IFACE=192.168.123.99 bash play.sh
```

Expected timeline:

```
[init] parquet: .../Pull_the_tray_episode_000000.parquet
[init] 1433 frames, first lower-8 dof = [ 0.0302 -0.1646  0.2018  0.7059 0. 0. 0. 0.]
[init] CYCLONEDDS_URI iface = 192.168.123.22
[2026-... INFO] Initialize G1_29_BodyController...
[2026-... INFO] [G1_29_ArmController] Waiting to subscribe dds...
... (DDS handshake, AMO init, Dex3 init -- ~5s) ...
[MAIN] standing warm-up 5.0s ...
[2026-... INFO] [maintain_standing] Stage C: AMO target lower=[...]
[feeder] starting, 1433 frames @ 30 Hz, loop=False
[MAIN] running. Ctrl+C to stop.
[CTRL] RTC startup smoothing engaged: first 50 ticks limited to 0.0200 rad/tick
[ARM] cmd14=[ 0.486 0.035 ...]
... (~48 s of replay for a 1433-frame episode) ...
[feeder] dataset exhausted at frame 1433; holding last action ...
[MAIN] feeder thread done -> initiating cooldown
[MAIN] returning to standing pose for 5.0s ...
[MAIN] shutdown complete
```

## Troubleshooting

### `ImportError: ... CXXABI_1.3.15 not found`

`conda activate` does not prepend `$CONDA_PREFIX/lib` to
`LD_LIBRARY_PATH`, but casadi 3.7's `libcasadi.so.3.7` needs the new
libstdc++ that ships in the conda env. `play.sh` fixes this with:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

If you run `python play.py` directly without `play.sh`, set this yourself.

### `[G1_29_ArmController] Waiting to subscribe dds...` hangs forever

DDS isn't seeing any G1 packets.
- Check `ip a` shows your interface has IP `192.168.123.X/24`
- Check the cable to G1
- Override iface: `PSI_DDS_IFACE=<your iface ip> bash play.sh`

### `conda env list` shows env but `conda activate g1_amo_play` fails

Make sure you sourced conda first:
```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```
or whatever path `conda info --base` gives you.

### `[CTRL] loop error: ...` mid-replay

Look at the WAIST_LOG csv (`/tmp/waist_log_<ts>.csv`). Each row carries
`tag` (`stand_B`/`stand_C_amo_target`/`stand_D`/`rtc`), `motorstate`,
`cmd_q`, `tau_est`, `torso_*`, `vx/vy/vyaw`, `have_vla`, `hand_state`,
`hand_cmd`. Diff the failing row against the one before to localize the
trigger.

## Design notes

- `play.py` is a copy of the upstream tested codepath with two surgical
  changes vs. [Psi0/real/deploy/psi-inference_rtc.py](https://github.com/physical-superintelligence-lab/Psi0/blob/main/real/deploy/psi-inference_rtc.py):
  1. The `RTCIPCClient` (`/tmp/psi0_rtc.sock` Unix socket) is gone — the
     parquet feeder writes `pred_action_buffer` directly in-process.
  2. The `RSCamera` (and the obs-send thread that fed it to the VLA) is
     gone — we don't need camera frames because we are not predicting.
  Everything else (RobotTaskmaster init, maintain_standing 4-stage warmup,
  60 Hz control loop, apply_action_from_buffer with AMO + arm/hand
  override, RTC 50-tick smoothing, WAIST_LOG) is verbatim.
- The CYCLONEDDS_URI XML is set in `play.py` via `os.environ.setdefault`
  *before* importing cyclonedds / unitree_sdk2py, so it always wins
  over a missing or stale env. Override via `PSI_DDS_IFACE`.
- `assets/g1/meshes/` is a superset of what `teleop/meshes/` was; we
  drop the duplicate. URDFs reference meshes by relative `meshes/...`
  path, resolved against the URDF's directory.

## Replacing the action source (Noitom etc.)

The parquet-feeder thread is the entire seam. Drop in:

```python
def noitom_feeder():
    sdk = noitom.connect(...)
    while running.is_set():
        head, lp, rp, lk, rk, joy = sdk.read()
        arm = master.body_ik.solve_arm_ik(lp, rp, ...)            # 14
        lq, rq = hand_retargeting(lk), hand_retargeting(rk)        # 14
        rpyh = lower_ik.solve(head, lp, rp, ...)                   # 4
        v = (joy.x, joy.y, joy.rz, joy.target_yaw)                 # 4
        action = np.concatenate([lq, rq, arm, rpyh, *v])           # 36
        with pred_action_lock:
            pred_action_buffer["actions"] = action[None, :]
        time.sleep(...)
```

The CasADi `solve_arm_ik` and dex_retargeting are already vendored and
working; only the input source changes.

## Provenance

- `teleop/`, `assets/g1/`, `play.py`'s control loop:
  [physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0)
  (Apache 2.0)
- `unitree_sdk2_python/`:
  [Psi0's modified fork of unitreerobotics/unitree_sdk2_python](https://github.com/physical-superintelligence-lab/unitree_sdk2_python)
  (BSD-3-Clause)
- `teleop/robot_control/dex_retargeting/`:
  [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting)
  (MIT) — bundled with relative imports, not the pip package
- `teleop/master_whole_body.py` 4-stage standing logic and
  `teleop/waist_logger.py`: written by the Psi0 deploy team
- AMO model (`teleop/amo_jit.pt` + adapter): trained by the Psi0 team
  (see [OpenTeleVision/AMO](https://github.com/OpenTeleVision/AMO) for
  the underlying architecture)
