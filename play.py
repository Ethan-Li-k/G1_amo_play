"""Single-terminal G1 dataset replay.

Reads a LeRobot v2.1 parquet episode and drives the real Unitree G1
through the AMO + arm-override + Dex3-override pipeline (same code path
as Psi0/real/deploy/psi-inference_rtc.py, just sourcing actions from a
file instead of a VLA model over IPC).

Usage:
    # via the wrapper (auto-detect conda + sane defaults):
    bash play.sh

    # direct (requires the env to already be active and LD_LIBRARY_PATH set):
    python play.py --parquet examples/Pull_the_tray_episode_000000.parquet

Env-var overrides:
    PSI_DDS_IFACE       network interface that talks to G1 (default 192.168.123.22)
    PSI_RTC_SMOOTH_TICKS first-N RTC ticks to rate-limit lower body (default 50)
    PSI_RTC_SMOOTH_MAX_STEP rate-limit per tick in rad (default 0.02)
    PSI_LOG_ARM_ACTION  '1' to print [ARM] cmd14 every 30 ticks (default '1')
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from multiprocessing import Array, Event

import numpy as np


# ----------------------------------------------------------------------
# Path & DDS env BEFORE any import of cyclonedds / unitree_sdk2py
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEOP_DIR = os.path.join(REPO_ROOT, "teleop")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TELEOP_DIR)
os.chdir(TELEOP_DIR)            # asset / jit relative paths assume this CWD

DDS_IFACE = os.environ.get("PSI_DDS_IFACE", "192.168.123.22")
os.environ.setdefault(
    "CYCLONEDDS_URI",
    "<CycloneDDS><Domain><General><NetworkInterfaceAddress>"
    + DDS_IFACE
    + "</NetworkInterfaceAddress></General></Domain></CycloneDDS>",
)

import pyarrow.parquet as pq

from teleop.master_whole_body import RobotTaskmaster
from teleop.robot_control.compute_tau import GetTauer
from teleop.waist_logger import WAIST_LOG


# ----------------------------------------------------------------------
# Constants -- mirror psi-inference_rtc.py
# ----------------------------------------------------------------------
FREQ_CTRL = 60                 # control loop rate
DATASET_FPS = 30.0             # parquet recording rate (info.json fps)
WARMUP_SECONDS = 5.0           # standing warm-up before control loop
COOLDOWN_SECONDS = 5.0         # standing hold after dataset ends


# ----------------------------------------------------------------------
# Shared buffers
# ----------------------------------------------------------------------
shared_data = {
    "kill_event": Event(),
    "session_start_event": Event(),
    "failure_event": Event(),
    "end_event": Event(),
    "dirname": None,
}
kill_event = shared_data["kill_event"]

robot_shm_array = Array("d", 512, lock=False)
teleop_shm_array = Array("d", 64, lock=False)

pred_action_buffer = {"actions": None}
pred_action_lock = threading.Lock()

running = Event()
running.set()


def load_actions(parquet_path: str) -> np.ndarray:
    table = pq.read_table(parquet_path)
    actions = np.array(table.column("action").to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 36:
        raise ValueError(
            f"action column shape {actions.shape}; expected (N, 36) per "
            f"Psi0 LeRobot v2.1 modality.json"
        )
    return actions


def parquet_feeder(actions: np.ndarray, loop: bool):
    """Push one dataset frame into pred_action_buffer every 1/DATASET_FPS s."""
    dt = 1.0 / DATASET_FPS
    N = len(actions)
    next_tick = time.perf_counter()
    i = 0
    last_printed = time.perf_counter()

    print(f"[feeder] starting, {N} frames @ {DATASET_FPS:.0f} Hz, loop={loop}")
    while running.is_set() and not kill_event.is_set():
        if i >= N:
            if loop:
                print(f"[feeder] dataset wrapped at {i}, looping")
                i = 0
            else:
                print(f"[feeder] dataset exhausted at frame {i}; "
                      f"holding last action (control loop keeps cmd)")
                return

        # Shape (1, 36) to match psi_serve_rtc-trainingtimertc payload.
        frame = actions[i].astype(np.float32)[None, :]
        with pred_action_lock:
            pred_action_buffer["actions"] = frame
        i += 1

        next_tick += dt
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        elif -sleep > 0.1:
            next_tick = time.perf_counter()
            print(f"[feeder] WARNING: drifted {-sleep*1e3:.0f} ms, "
                  f"resetting clock")

        if time.perf_counter() - last_printed > 5.0:
            print(f"[feeder] frame {i}/{N}")
            last_printed = time.perf_counter()


def build_apply_action_from_buffer(master, get_tauer, log_arm_action):
    """Closure -- equivalent to psi-inference_rtc.apply_action_from_buffer."""
    SMOOTH_TICKS = int(os.environ.get("PSI_RTC_SMOOTH_TICKS", "50"))
    SMOOTH_MAX_STEP = float(os.environ.get("PSI_RTC_SMOOTH_MAX_STEP", "0.02"))
    state = {"smooth_tick": 0, "arm_debug": 0}

    def apply_action_from_buffer(last_pd_target):
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        have_vla = False
        with pred_action_lock:
            action = pred_action_buffer["actions"]
            if action is not None:
                have_vla = True
                action = action[0]

        arm_cmd = None
        hand_cmd = None
        if have_vla:
            if action.shape[0] < 36:
                print("[CTRL] invalid action shape:", action.shape)
            else:
                vx = action[32]
                vy = action[33]
                vyaw = action[34]
                target_yaw = action[35]

                vx = 0.6 if vx > 0.25 else 0
                vy = 0 if abs(vy) < 0.3 else 0.5 * (1 if vy > 0 else -1)

                rpyh = action[28:32]
                arm_cmd = action[14:28]
                hand_cmd = action[:14]

                master.torso_roll = rpyh[0]
                master.torso_pitch = rpyh[1]
                master.torso_yaw = rpyh[2]
                master.torso_height = rpyh[3]
                master.vx = vx
                master.vy = vy
                master.vyaw = vyaw
                master.target_yaw = target_yaw

                master.prev_torso_roll = master.torso_roll
                master.prev_torso_pitch = master.torso_pitch
                master.prev_torso_yaw = master.torso_yaw
                master.prev_torso_height = master.torso_height
                master.prev_vx = master.vx
                master.prev_vy = master.vy
                master.prev_vyaw = master.vyaw
                master.prev_target_yaw = master.target_yaw
                master.prev_arm = arm_cmd
                master.prev_hand = hand_cmd

        if not have_vla:
            master.torso_roll = master.prev_torso_roll
            master.torso_pitch = master.prev_torso_pitch
            master.torso_yaw = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height
            master.vx = master.prev_vx
            master.vy = 0
            master.vyaw = master.prev_vyaw
            master.target_yaw = master.prev_target_yaw

        master.get_ik_observation(record=False)

        pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=master.observation,
            extra_hist=master.extra_hist,
            is_teleop=False,
        )

        master.last_action = np.concatenate(
            [
                raw_action.copy(),
                (master.motorstate - master.default_dof_pos)[15:] / master.action_scale,
            ]
        )

        if arm_cmd is not None:
            pd_target[15:] = arm_cmd
            tau_arm = np.asarray(get_tauer(arm_cmd), dtype=np.float64).reshape(-1)
            pd_tauff[15:] = tau_arm
            if log_arm_action and state["arm_debug"] % 30 == 0:
                print(
                    "[ARM] cmd14="
                    + np.array2string(arm_cmd, precision=3, suppress_small=True)
                )
            state["arm_debug"] += 1

        if hand_cmd is not None:
            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_cmd

        # RTC startup smoothing (first 50 ticks)
        if state["smooth_tick"] < SMOOTH_TICKS:
            ref = (
                last_pd_target[:15] if last_pd_target is not None else master.motorstate[:15]
            )
            ref = np.asarray(ref, dtype=np.float64)
            delta = pd_target[:15] - ref
            delta_max = float(np.max(np.abs(delta)))
            if delta_max > SMOOTH_MAX_STEP:
                scale = SMOOTH_MAX_STEP / delta_max
                pd_target[:15] = ref + delta * scale
                try:
                    new_tauff = master.body_ik.compute_whole_body_tau(pd_target)
                    pd_tauff[:15] = new_tauff[:15]
                except Exception as e:
                    print(f"[CTRL] smooth tauff recompute failed: {e}")
            if state["smooth_tick"] == 0:
                print(
                    f"[CTRL] RTC startup smoothing engaged: first "
                    f"{SMOOTH_TICKS} ticks limited to {SMOOTH_MAX_STEP:.4f} rad/tick"
                )
            state["smooth_tick"] += 1
            if state["smooth_tick"] == SMOOTH_TICKS:
                print(f"[CTRL] RTC startup smoothing released after {SMOOTH_TICKS} ticks")

        master.body_ctrl.ctrl_whole_body(
            pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15]
        )

        try:
            tau_est = master.body_ctrl.get_current_motor_tau_est()
        except Exception:
            tau_est = None
        WAIST_LOG.log(
            tag="rtc",
            motorstate=master.motorstate,
            cmd_q=pd_target,
            velstate=master.velstate,
            tau_est=tau_est,
            torso_rpy=(master.torso_roll, master.torso_pitch, master.torso_yaw),
            torso_h=master.torso_height,
            vxyyaw=(master.vx, master.vy, master.vyaw),
            have_vla=have_vla,
            hand_state=getattr(master, "handstate", None),
            hand_cmd=hand_cmd if hand_cmd is not None else master.hand_shm_array,
        )

        return pd_target

    return apply_action_from_buffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        required=True,
        help="LeRobot v2.1 parquet episode (36-d action column).",
    )
    ap.add_argument("--loop", action="store_true",
                    help="Loop the dataset instead of stopping at EOF.")
    args = ap.parse_args()

    parquet_path = os.path.abspath(args.parquet)
    print(f"[init] parquet: {parquet_path}")
    actions = load_actions(parquet_path)
    print(f"[init] {len(actions)} frames, first lower-8 dof = "
          f"{actions[0, 28:36].round(4)}")
    print(f"[init] CYCLONEDDS_URI iface = {DDS_IFACE} "
          f"(override with env PSI_DDS_IFACE)")

    master = RobotTaskmaster(
        task_name="parquet_replay",
        shared_data=shared_data,
        robot_shm_array=robot_shm_array,
        teleop_shm_array=teleop_shm_array,
        robot="g1",
    )
    get_tauer = GetTauer()
    log_arm_action = os.environ.get("PSI_LOG_ARM_ACTION", "1") == "1"
    apply_action_from_buffer = build_apply_action_from_buffer(
        master, get_tauer, log_arm_action
    )

    def control_loop():
        dt = 1.0 / FREQ_CTRL
        last_pd_target = None
        while running.is_set() and not kill_event.is_set():
            try:
                last_pd_target = apply_action_from_buffer(last_pd_target)
            except Exception as e:
                print("[CTRL] loop error:", e)
            time.sleep(dt)
        print("[CTRL] control loop stopped")

    try:
        stabilize = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize.start()
        master.episode_kill_event.set()
        print(f"[MAIN] standing warm-up {WARMUP_SECONDS}s ...")
        time.sleep(WARMUP_SECONDS)
        master.episode_kill_event.clear()
        master.reset_yaw_offset = True

        t_ctrl = threading.Thread(target=control_loop, daemon=True)
        t_ctrl.start()
        t_feed = threading.Thread(
            target=parquet_feeder, args=(actions, args.loop), daemon=True
        )
        t_feed.start()
        print("[MAIN] running. Ctrl+C to stop.")

        while not kill_event.is_set() and running.is_set():
            time.sleep(0.5)
            if not t_feed.is_alive() and not args.loop:
                print("[MAIN] feeder thread done -> initiating cooldown")
                break

        running.clear()
        master.episode_kill_event.set()
        print(f"[MAIN] returning to standing pose for {COOLDOWN_SECONDS}s ...")
        time.sleep(COOLDOWN_SECONDS)
        master.episode_kill_event.clear()

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl-C")
        running.clear()
        kill_event.set()
    finally:
        shared_data["end_event"].set()
        try:
            master.stop()
        except Exception as e:
            print(f"[MAIN] master.stop() error: {e}")
        print("[MAIN] shutdown complete")


if __name__ == "__main__":
    main()
