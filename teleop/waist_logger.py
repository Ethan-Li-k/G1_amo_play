"""Lower-body joint debug logger (debug-only, to diagnose waist/thigh twitch).

Records state q, commanded q, velocity dq and tau_est for these G1 joints:
  0 left_hip_pitch   1 left_hip_roll   2 left_hip_yaw   3 left_knee
  6 right_hip_pitch  7 right_hip_roll  8 right_hip_yaw  9 right_knee
  12 waist_yaw       13 waist_roll     14 waist_pitch
plus context: torso setpoints (roll/pitch/yaw/height), vx/vy/vyaw, have_vla.

CSV is written on process exit (atexit) to
  $PSI_WAIST_LOG_DIR/waist_log_<ts>_<pid>.csv   (default /tmp)
"""
import atexit
import os
import threading
import time
from pathlib import Path

import numpy as np

JOINT_IDX = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14]
JOINT_NAMES = [
    "lhip_p", "lhip_r", "lhip_y", "lknee",
    "rhip_p", "rhip_r", "rhip_y", "rknee",
    "waist_y", "waist_r", "waist_p",
]


def _pick(a):
    if a is None:
        return [float("nan")] * len(JOINT_IDX)
    a = np.asarray(a).reshape(-1)
    out = []
    n = a.shape[0]
    for i in JOINT_IDX:
        out.append(float(a[i]) if i < n else float("nan"))
    return out


def _build_header():
    h = ["t", "tag"]
    for kind in ("state", "cmd", "vel", "tau"):
        for n in JOINT_NAMES:
            h.append(f"{kind}_{n}")
    h += [
        "torso_roll_sp", "torso_pitch_sp", "torso_yaw_sp", "torso_height_sp",
        "vx", "vy", "vyaw", "have_vla",
    ]
    # 14 hand-state q values then 14 hand-cmd q values. Order is
    # [left thumb..pinky 7-DOF] then [right thumb..pinky 7-DOF],
    # matching hand_shm_array[0:7]+hand_shm_array[7:14].
    for i in range(14):
        h.append(f"hand_state_{i}")
    for i in range(14):
        h.append(f"hand_cmd_{i}")
    return h


_HEADER = _build_header()


class WaistLogger:
    def __init__(self, max_rows=1_000_000):
        self._lock = threading.Lock()
        self._rows = []
        self._max = max_rows
        self._dumped = False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(os.environ.get("PSI_WAIST_LOG_DIR", "/tmp"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"waist_log_{stamp}_{os.getpid()}.csv"
        atexit.register(self.dump)
        print(f"[WAIST_LOG] recording to {self.path}")

    def log(self, tag, motorstate, cmd_q, velstate=None, tau_est=None,
            torso_rpy=None, torso_h=None, vxyyaw=None, have_vla=None,
            hand_state=None, hand_cmd=None):
        vals = _pick(motorstate) + _pick(cmd_q) + _pick(velstate) + _pick(tau_est)
        if torso_rpy is None:
            tr = [float("nan")] * 3
        else:
            tr = [float(torso_rpy[0]), float(torso_rpy[1]), float(torso_rpy[2])]
        th = float("nan") if torso_h is None else float(torso_h)
        if vxyyaw is None:
            vv = [float("nan")] * 3
        else:
            vv = [float(vxyyaw[0]), float(vxyyaw[1]), float(vxyyaw[2])]
        hv = -1 if have_vla is None else int(bool(have_vla))

        def _pick14(a):
            if a is None:
                return [float("nan")] * 14
            arr = np.asarray(a).reshape(-1)
            out = [float(arr[i]) if i < arr.shape[0] else float("nan") for i in range(14)]
            return out
        hs = _pick14(hand_state)
        hc = _pick14(hand_cmd)
        row = [time.time(), tag] + vals + tr + [th] + vv + [hv] + hs + hc
        with self._lock:
            if len(self._rows) < self._max:
                self._rows.append(row)

    def dump(self):
        with self._lock:
            if self._dumped:
                return
            self._dumped = True
            rows = list(self._rows)
        if not rows:
            print("[WAIST_LOG] no rows captured")
            return
        try:
            with open(self.path, "w") as f:
                f.write(",".join(_HEADER) + "\n")
                for r in rows:
                    f.write(f"{r[0]:.6f},{r[1]}")
                    for x in r[2:]:
                        if isinstance(x, float):
                            f.write(f",{x:.6f}")
                        else:
                            f.write(f",{x}")
                    f.write("\n")
            print(f"[WAIST_LOG] dumped {len(rows)} rows to {self.path}")
        except Exception as e:
            print(f"[WAIST_LOG] dump failed: {e}")


WAIST_LOG = WaistLogger()
