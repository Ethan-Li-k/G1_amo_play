import os
import sys
import threading
import time
import traceback
from collections import deque
from multiprocessing import (Array, Event, Lock, Manager, Process, Queue,
                             shared_memory)
import csv
import mujoco
import numpy as np
import torch
from teleop.merger import DataMerger
from teleop.robot_control.robot_body import G1_29_BodyController
from teleop.robot_control.robot_body_ik import G1_29_BodyIK
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
from teleop.utils.logger import logger
from teleop.writers import IKDataWriter
from teleop.robot_control.compute_tau import GetTauer
from teleop.waist_logger import WAIST_LOG

from scipy.spatial.transform import Rotation


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.constants import *

CONTROL_DELAY = 1.0 / 60


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec


class RobotTaskmaster:
    def __init__(
        self, task_name, shared_data, robot_shm_array, teleop_shm_array, robot="h1"
    ): 
        
        self.get_tauer = GetTauer()

        self.task_name = task_name
        self.robot = robot

        self.shared_data = shared_data
        self.episode_kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.failure_event = shared_data["failure_event"]  # TODO: redundent
        self.end_event = shared_data["end_event"]  # TODO: redundent

        self.robot_shm_array = robot_shm_array
        self.teleop_shm_array = teleop_shm_array

        self.teleop_lock = Lock()

        # Controller parameters
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        # AMO parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05

        self.nj = 29
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15 # no wrist joint (model input)
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8 # 4+4 no wrist joint

        self.default_dof_pos = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0,
                0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0,
            ])
        
        self.motorstate = np.zeros(self.nj, dtype=np.float32)
        self.velstate = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)
        self.action_scale = 0.25
        
        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[np.r_[15:19, 22:26]]
        self.demo_obs_template[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75
        
        self.target_yaw = 0.0 
        self.reset_yaw_offset = True
        self.yaw_offset = 0.0
        self.dyaw = 0.0

        self.dt = 0.02

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3
        self.control_dt = 0.02

        # self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        # if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
        #     self.gait_cycle = np.array([0.25, 0.25])
        # if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
        #     self.gait_cycle = np.array([0.25, 0.75])

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))
        
        self.adapter = torch.jit.load("adapter_jit.pt", map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False
        
        norm_stats = torch.load("adapter_norm_stats.pt", weights_only=False)
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        # self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        # initialize parameters for torso
        self.torso_height = 0.75
        self.torso_roll = 0.0
        self.torso_pitch = 0.0
        self.torso_yaw = 0.0

        self.prev_torso_roll   = 0.0
        self.prev_torso_pitch  = 0.0
        self.prev_torso_yaw    = 0.0
        self.prev_torso_height = 0.75
        self.prev_arm = None
        self.prev_hand = None

        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_vyaw = 0.0
        self.prev_dyaw = 0.0
        self.prev_target_yaw = 0.0

        # self.tau_history = []
        # self.tau_log_path = f"tau_log_{int(time.time())}.csv"
        # self.tau_file = open(self.tau_log_path, mode='w', newline='')
        # self.tau_writer = csv.writer(self.tau_file)
        
        # self.tau_writer.writerow(['t', 'tau_22', 'tau_23', 'tau_24', 'tau_25', 'tau_26', 'tau_27', 'tau_28'])

        # self.start_time = time.time()
        
        try:
            if robot == "g1":
                logger.info("Using g1 controllers")
                self.body_ctrl = G1_29_BodyController()
                print("body_ctrl ok!")
                self.body_ik = G1_29_BodyIK(Visualization=False)
                print("body_ik ok!")
                self.dual_hand_data_lock = Lock()
                dual_hand_state_array = Array(
                    "d", 14, lock=False
                )  # [output] current left, right hand state(14) data.
                dual_hand_action_array = Array(
                    "d", 14, lock=False
                )  # [output] current left, right hand action(14) data.
                self.hand_shm = shared_memory.SharedMemory(
                    create=True, size=14 * np.dtype(np.float64).itemsize
                )
                self.hand_shm_array = np.ndarray(
                    (14,), dtype=np.float64, buffer=self.hand_shm.buf
                )

                self.hand_ctrl = Dex3_1_Controller(
                    self.hand_shm_array,
                    self.dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                )
            else:
                logger.error("unknown robot")
                exit(-1)
        except Exception as e:
            logger.error(f"Master: failed initalizing controllers/ik_solvers: {e}")
            traceback.print_exc()
            logger.error(f"Master: exiting")
            exit(-1)

        self.first = True
        self.merge_proc = None
        self.ik_writer = None
        self.running = False
        # self.h1_lock = Lock()
        self._idx = 0
    
    

    def safelySetMotor(self, sol_q, last_sol_q, tau_ff):
        """
        Per-tick max joint step (PSI_MAX_JOINT_STEP, default 0.05 rad) plus hard reject
        if IK jump exceeds dynamic_thresholds vs last *commanded* pose.

        Returns (ok, commanded_q). On success, callers must set last_pd_target = commanded_q,
        not raw IK output — otherwise the next delta is wrong and the robot can jerk.
        """
        sol_q = np.asarray(sol_q, dtype=np.float64).copy()
        tau_ff = np.asarray(tau_ff, dtype=np.float64)
        motor = np.asarray(self.motorstate, dtype=np.float64)
        max_step_per_tick = float(os.environ.get("PSI_MAX_JOINT_STEP", "0.05"))

        arm_q_poseList = sol_q[15:]
        arm_q_tau_ff = tau_ff[15:]
        lower_q_poseList = sol_q[:15]
        lower_q_tau_ff = tau_ff[:15]
        dynamic_thresholds = np.array(
            [np.pi / 2] * 15
            + [np.pi / 3] * 5
            + [np.pi] * 2
            + [np.pi / 3] * 5
            + [np.pi] * 2
        )

        if last_sol_q is None:
            delta0 = sol_q - motor
            delta_max0 = float(np.max(np.abs(delta0)))
            if delta_max0 > 0.5:
                logger.warning(
                    "[safelySetMotor] FIRST CALL large delta vs motorstate: %.3f rad (e-stop if unexpected)",
                    delta_max0,
                )
            if delta_max0 > max_step_per_tick:
                sol_q = motor + delta0 * (max_step_per_tick / delta_max0)
                arm_q_poseList = sol_q[15:]
                lower_q_poseList = sol_q[:15]
            self.body_ctrl.ctrl_whole_body(
                arm_q_poseList, arm_q_tau_ff, lower_q_poseList, lower_q_tau_ff, True
            )
            return True, sol_q.copy()

        last_sol_q = np.asarray(last_sol_q, dtype=np.float64)
        delta = sol_q - last_sol_q
        if np.any(np.abs(delta) > dynamic_thresholds):
            logger.error(
                "Master: ik movement too large (max elementwise delta over threshold)! refusing this tick"
            )
            return False, None

        delta_max = float(np.max(np.abs(delta)))
        if delta_max > max_step_per_tick:
            scale = max_step_per_tick / delta_max
            sol_q = last_sol_q + delta * scale
            arm_q_poseList = sol_q[15:]
            lower_q_poseList = sol_q[:15]
            logger.debug(
                "[safelySetMotor] clipped delta from %.3f to %.3f rad (scale=%.2f)",
                delta_max,
                max_step_per_tick,
                scale,
            )

        self.body_ctrl.ctrl_whole_body(
            arm_q_poseList, arm_q_tau_ff, lower_q_poseList, lower_q_tau_ff
        )
        return True, sol_q.copy()

    def setHandMotors(self, right_qpos, left_qpos):
        if right_qpos is not None and left_qpos is not None:
            right_hand_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
            right_hand_angles.append(1.2 - right_qpos[8])
            right_hand_angles.append(0.5 - right_qpos[9])

            left_hand_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
            left_hand_angles.append(1.2 - left_qpos[8])
            left_hand_angles.append(0.5 - left_qpos[9])
            self.hand_ctrl.ctrl_dual_hand(right_hand_angles, left_hand_angles)
            # self.left_hand_array[:] = left_qpos
            # self.right_hand_array[:] = right_qpos
            # self.hand_ctrl.ctrl_dual_hand(right_qpos, left_qpos)
        return left_qpos, right_qpos

    def start(self):
        # logger.debug(f"Master: Process ID (PID) {os.getpid()}")
        try:
            stabilize_thread = threading.Thread(target=self.maintain_standing, daemon=True)
            self.reset_yaw_offset = True 
            stabilize_thread.start()
            while not self.end_event.is_set():
                logger.info("Master: waiting to start")
                self.session_start_event.wait() # print s to teleop
                logger.info(
                    "Master: start event recvd. clearing start event. starting session"
                )
                self.reset_yaw_offset = True
                self.run_session()
                logger.debug("Master: merging data...")
                if not self.failure_event.is_set():
                    self.merge_data()  # TODO: maybe a separate thread?
                    logger.info("Master: merge finished. Preparing for a new run...")
                else:
                    # self.delete_last_data()
                    logger.info(
                        "Master: not merging. Preparing for a new run to override..."
                    )
                self.reset()
                logger.info("Master: reset finished")
        finally:
            self.stop()

            if self.robot == "g1":
                self.hand_shm.close()
                self.hand_shm.unlink()
            logger.info("Master: exited")



    def get_ik_observation(self, record=True):
        rpy = self.rpy
        
        if record:
            self.target_yaw += self.vyaw * self.dt

            dyaw = rpy[2] - self.yaw_offset - self.target_yaw
            dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi


            if self._in_place_stand_flag:
                dyaw = 0.0

            self.dyaw = dyaw
        
        else:
            dyaw = rpy[2] - self.yaw_offset - self.target_yaw
            dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
            if self._in_place_stand_flag:
                dyaw = 0.0

            self.dyaw = dyaw


        obs_idx = np.r_[0:19, 22:26] 
        obs_dof_vel = self.velstate[obs_idx]
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        obs_dof_pos = self.motorstate[obs_idx]
        obs_default_dof_pos = self.default_dof_pos[obs_idx]

        obs_last_action = self.last_action[obs_idx]

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        adapter_input_np = np.concatenate([np.zeros(4), obs_dof_pos[15:]])

        adapter_input_np[0] = self.torso_height
        adapter_input_np[1] = self.torso_yaw
        adapter_input_np[2] = self.torso_pitch
        adapter_input_np[3] = self.torso_roll

        self.adapter_input = torch.tensor(adapter_input_np).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
                    self.ang_vel * self.scales_ang_vel,
                    rpy[:2],
                    (np.sin(self.dyaw),
                    np.cos(self.dyaw)),
                    (obs_dof_pos - obs_default_dof_pos),
                    obs_dof_vel * self.scales_dof_vel,
                    obs_last_action,
                    gait_obs,
                    self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = obs_dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.vx
        obs_demo[self._n_demo_dof+1] = self.vy
        self._in_place_stand_flag = (np.abs(self.vx) < 0.1) and (np.abs(self.vy) < 0.1) and (np.abs(self.vyaw) < 0.1)

        obs_demo[self._n_demo_dof+3] = self.torso_yaw
        obs_demo[self._n_demo_dof+4] = self.torso_pitch
        obs_demo[self._n_demo_dof+5] = self.torso_roll
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = self.torso_height

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        self.observation = np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))
        self.extra_hist = self.extra_history_buf

        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.75])

        return self.observation, self.extra_hist

        




    def get_robot_data(self):
        motorstate = self.body_ctrl.get_current_motor_q()
        self.motorstate = motorstate
        velstate = self.body_ctrl.get_current_motor_dq()
        self.velstate = velstate
        logger.debug(f"motorstate f{motorstate}")

        # taustate = self.body_ctrl.get_current_motor_tau_est()
    
        # last_seven = taustate[-7:]
        
        # timestamp = time.time() - self.start_time
        # self.tau_writer.writerow([timestamp] + list(last_seven))

        controllerstate = self.body_ctrl.remote_controller
        lx = controllerstate.lx
        ly = controllerstate.ly
        rx = controllerstate.rx
        ry = controllerstate.ry
        buttons = controllerstate.button
        # print(f"Left stick: ({lx:.2f}, {ly:.2f}), Right stick: ({rx:.2f}, {ry:.2f})")
        if buttons[3]:  # KeyMap.A
            logger.warning("[E-STOP] Emergency stop button pressed! Triggering shutdown...")
            self.end_event.set()
            self.session_start_event.set()
            self.episode_kill_event.set()

        def scale_vx(v):
            return 0 if abs(v) < 0.3 else 0.6 * (1 if v > 0 else -1)

        def scale_vy(v):
            # return 0 if abs(v) < 0.3 else 0.35 * (1 if v > 0 else -1)
            return 0 if abs(v) < 0.7 else 0.5 * (1 if v > 0 else -1)


        # --- vy & vyaw: 0 / ±0.25 ---
        def scale_vyaw(v):
            if abs(v) < 0.2:
                return 0
            return (0.3 if abs(v) < 0.5 else 0.5) * (1 if v > 0 else -1)
            # return 0 if abs(v) < 0.5 else 0.5 * (1 if v > 0 else -1)

        # apply mapping
        self.vx = scale_vx(ly)
        self.vy = scale_vy(-lx)
        self.vyaw = scale_vyaw(-rx)

        # self.target_yaw += self.vyaw * self.dt

        # print("self.yaw:", self.yaw)

        # print("in_place_stand_flag:", self._in_place_stand_flag)

        handstate = self.hand_ctrl.get_current_dual_hand_q()
        self.handstate = handstate
        if self.robot == "g1":
            hand_press_state = self.hand_ctrl.get_current_dual_hand_pressure()
            robot_sizes = G1_sizes

        imustate = self.body_ctrl.get_imu_data()
        self.imustate = imustate
        self.quat = np.array(imustate.quaternion, dtype=np.float32)
        self.imu_rpy = np.array(imustate.rpy, dtype=np.float32)
        self.rpy = quatToEuler(self.quat)
        # print("robot_yaw:", self.rpy[2])

        imu_yaw = self.rpy[2]

        if self.reset_yaw_offset:
            self.yaw_offset = imu_yaw
            self.reset_yaw_offset = False 

        self.ang_vel = np.array(imustate.gyroscope, dtype=np.float32)

        odomstate = self.body_ctrl.get_odom_data()
        self.odomstate = odomstate
        self.odom_pos = odomstate["position"]
        self.odom_vel = odomstate["velocity"]

        # self.torso_height = self.odom_pos[2]
        # self.torso_roll = self.rpy[0]
        # self.torso_pitch = self.rpy[1]
        # self.torso_yaw = self.rpy[2]


        # var_imu = dir(imustate)
        current_lr_arm_q = self.body_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = self.body_ctrl.get_current_dual_arm_dq()

        motor_state_size = robot_sizes.ARM_STATE_SIZE + robot_sizes.LEG_STATE_SIZE
        # with self.h1_lock:
        motor_start = 0
        hand_start = motor_start + motor_state_size
        quat_start = hand_start + robot_sizes.HAND_STATE_SIZE
        accel_start = quat_start + robot_sizes.IMU_QUATERNION_SIZE
        gyro_start = accel_start + robot_sizes.IMU_ACCELEROMETER_SIZE
        rpy_start = gyro_start + robot_sizes.IMU_GYROSCOPE_SIZE
        pos_start = rpy_start + robot_sizes.IMU_RPY_SIZE
        velocity_start = pos_start + robot_sizes.ODOM_POSITION_SIZE
        odom_rpy_start = velocity_start + robot_sizes.ODOM_VELOCITY_SIZE
        odom_quat_start = odom_rpy_start + robot_sizes.ODOM_RPY_SIZE
        odom_quat_end = odom_quat_start + robot_sizes.ODOM_QUATERNION_SIZE

        self.robot_shm_array[motor_start:hand_start] = motorstate[0:motor_state_size]
        self.robot_shm_array[hand_start:quat_start] = handstate
        self.robot_shm_array[quat_start:accel_start] = imustate.quaternion
        self.robot_shm_array[accel_start:gyro_start] = imustate.accelerometer
        self.robot_shm_array[gyro_start:rpy_start] = imustate.gyroscope
        self.robot_shm_array[rpy_start:pos_start] = imustate.rpy

        # self.robot_shm_array[pos_start:velocity_start] = odomstate["position"]
        # self.robot_shm_array[velocity_start:odom_rpy_start] = odomstate["velocity"]
        # self.robot_shm_array[odom_rpy_start:odom_quat_start] = odomstate["orientation_rpy"]
        # self.robot_shm_array[odom_quat_start:odom_quat_end] = odomstate["orientation_quaternion"]

        if self.robot == "g1":
            # press_start = rpy_start + robot_sizes.IMU_RPY_SIZE
            # self.robot_shm_array[rpy_start:press_start] = imustate.rpy
            self.robot_shm_array[pos_start:velocity_start] = odomstate["position"]
            self.robot_shm_array[velocity_start:odom_rpy_start] = odomstate["velocity"]
            self.robot_shm_array[odom_rpy_start:odom_quat_start] = odomstate[
                "orientation_rpy"
            ]
            self.robot_shm_array[odom_quat_start:odom_quat_end] = odomstate[
                "orientation_quaternion"
            ]

            self.robot_shm_array[
                odom_quat_end : odom_quat_end + robot_sizes.HAND_PRESS_SIZE
            ] = hand_press_state.flatten()

        # elif self.robot == "h1":
        #     self.robot_shm_array[rpy_start:] = imustate.rpy

        return current_lr_arm_q, current_lr_arm_dq

    def get_teleoperator_data(self):
        with self.teleop_lock:
            teleop_data = self.teleop_shm_array.copy()
        # logger.debug(f"Master: receving data : {teleop_data}")
        if np.all(teleop_data == 0):
            logger.debug(f"Master: not receving data yet: {teleop_data}")
            return False, None, None, None, None, None
        head_rmat = teleop_data[0:16].reshape(4, 4)
        left_pose = teleop_data[16:32].reshape(4, 4)
        right_pose = teleop_data[32:48].reshape(4, 4)
        left_qpos = teleop_data[48:55]
        right_qpos = teleop_data[55:62]
        return True, head_rmat, left_pose, right_pose, left_qpos, right_qpos

    def _session_init(self):
        if "dirname" not in self.shared_data:
            logger.error("Master: failed to get dirname")
            exit(-1)
        self.running = True
        self.ik_writer = IKDataWriter(self.shared_data["dirname"])
        logger.debug("Master: getting teleop shm name")


    def _prefill_stand_observation_history(self, n_ticks=None):
        """DEPRECATED in static-stand mode. Kept as a no-op helper in case
        someone later wants to warm up policy history (set PSI_STAND_STATIC=0).
        Populates proprio/extra history with an obs computed from the current
        motorstate.
        """
        if n_ticks is None:
            try:
                n_ticks = int(os.environ.get("PSI_STAND_PREFILL_TICKS",
                                             str(max(self.history_len, self.extra_history_len))))
            except Exception:
                n_ticks = max(self.history_len, self.extra_history_len)
        try:
            self.get_robot_data()
        except Exception:
            pass
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.torso_height = 0.75
        self.torso_roll = 0.0
        self.torso_pitch = 0.0
        self.torso_yaw = 0.0
        for _ in range(n_ticks):
            try:
                self.get_ik_observation(record=False)
            except Exception:
                break

    def maintain_standing(self):
        logger.info("Master: Entering pre-run stabilization loop (maintain_standing)...")
        self.episode_kill_event.set()
        # Pre-run static stand. Active only while episode_kill_event is set,
        # i.e. the 30s pre-RTC window plus the 5s post-RTC tail. RTC/VLA
        # control loop (apply_action_from_buffer / ctrl_whole_body) is not
        # touched.
        #
        # Four-stage design:
        #
        # Stage A: snapshot motorstate as hold_pose, precompute hold_tauff
        #   (gravity comp at the snapshot pose).
        # Stage B (~hold_ticks ticks): cmd ≡ hold_pose, AMO NOT called.
        #   PD with full gravity comp keeps the robot exactly at the
        #   snapshot. Each tick we still call get_ik_observation so the
        #   proprio/extra history fills with quiet "standing at hold_pose"
        #   frames. By the end of Stage B the LSTM input is clean: no IMU
        #   warm-up jitter, no zero-history, no operator hand-off transient.
        # Stage C: single AMO call. With Stage B's quiet history feeding
        #   the LSTM, AMO's output is its symmetric, settled stand pose
        #   target — pd_target_static.
        # Stage D: rate-limited slow ramp from cmd to pd_target_static
        #   (default 0.003 rad/tick ≈ 0.15 rad/s). After the ramp completes
        #   the cmd is frozen at pd_target_static for the rest of the
        #   stand window. tauff is recomputed each tick to match the
        #   current cmd pose.
        #
        # Why this eliminates kicking:
        # - Stage B is dead silence: no AMO, no cmd motion. Source of all
        #   prior ramp/lift kicks is gone for the first ~1.5s.
        # - Stage C runs AMO once on a settled history, so its output is
        #   not asymmetric from IMU/LSTM warm-up. Both legs get the same
        #   target.
        # - Stage D's ramp rate is well below the PD natural frequency
        #   (≈3Hz). PD tracks the cmd glide without overshoot, so the
        #   leg slides smoothly from hold_pose into pd_target_static
        #   instead of being jerked.
        # - At t=30s, RTC takes over with the robot already sitting at
        #   AMO's preferred stand pose. AMO's first per-tick output is
        #   self-consistent — no rectification kick at the handoff.
        #
        # Why this is safe vs the old "anchor to motorstate" bug:
        # - In Stage B, cmd is a CONSTANT snapshot, not live motorstate.
        #   When the robot drifts off-snapshot under disturbance, PD
        #   error grows and pulls it back. hold_tauff is the gravity comp
        #   for the snapshot pose, so the steady-state torque load is
        #   carried by ff and PD only handles deviations.
        # - In Stage D, target is pd_target_static (≠ live motorstate).
        #   PD error is non-zero throughout the ramp.
        #
        # Handoff to RTC/VLA:
        # - last_action is set to AMO's raw_action_static during Stage C
        #   so the obs history is action-coherent. RTC's first tick
        #   overwrites it.
        # - history buffers are continuously refreshed by get_ik_observation
        #   in every stage, so the handoff sees a clean recent history.
        # - solve_whole_body_ik / safelySetMotor / episode_kill_event
        #   timing are unchanged. AMO and VLA paths are untouched.
        gait_lock = os.environ.get("PSI_STAND_GAIT_LOCK", "1") == "1"
        try:
            # Stage B duration in ticks. 10 ticks ≈ 0.2 s — bare minimum
            # for the IMU to spit out a few clean frames. Stage B is
            # "soft" by definition (cmd = motorstate, PD error = 0, only
            # ff supports against gravity), so we want this period as
            # short as possible. Stage D below gives the entire body real
            # PD authority by ramping every joint toward AMO's stand pose,
            # which is the same mechanism the original AMO-every-tick
            # design used to actively drive the robot stiff.
            hold_ticks = int(os.environ.get("PSI_STAND_HOLD_TICKS", "10"))
        except Exception:
            hold_ticks = 10
        hold_ticks = max(1, hold_ticks)
        try:
            # Stage D rate limit (rad/tick). 0.02 rad/tick ≈ 1.0 rad/s
            # before EMA, ≈ 0.5 rad/s effective. For a typical 0.8 rad
            # knee extension from the operator-placed deep squat to AMO's
            # preferred stand pose, the lift completes in ~1.6 s. Faster
            # than the previous 0.003 rad/tick (which took 5+ s and let
            # the user perceive "robot is not standing up"). Still well
            # below PD's natural frequency so motors track without
            # overshoot.
            lower_rate = float(os.environ.get("PSI_STAND_LOWER_RATE", "0.02"))
        except Exception:
            lower_rate = 0.02
        lower_rate = float(np.clip(lower_rate, 0.0005, 0.5))
        try:
            # Residual EMA after the rate-limit. alpha=1 disables EMA.
            lower_alpha = float(os.environ.get("PSI_STAND_LOWER_ALPHA", "0.5"))
        except Exception:
            lower_alpha = 0.5
        lower_alpha = float(np.clip(lower_alpha, 0.05, 1.0))
        # NOTE: an earlier revision implemented a "ff ramp from 0 to full
        # over the first N ticks" of Stage B, motivated by README
        # "首帧 standing 使用零 tau_ff". That made things WORSE on the real
        # G1: with ff_scale<1 a large fraction of the gravity load lands
        # on PD, the joint sags noticeably, and PD's overshoot recovery
        # produces the very transient the ramp was trying to avoid. CSV
        # 18:01:05 showed Stage B waist_p pp=0.087 rad during the ramp
        # but only 0.013 rad once ff_scale reached 1.0. So we always send
        # full hold_tauff from tick 1.
        waist_debug_every = int(os.environ.get("PSI_WAIST_DEBUG_EVERY", "60"))

        def _zero_torso_targets():
            self.vx = 0.0
            self.vy = 0.0
            self.vyaw = 0.0
            self.torso_height = 0.75
            self.torso_roll = 0.0
            self.torso_pitch = 0.0
            self.torso_yaw = 0.0

        stand_tick = 0
        while not self.end_event.is_set():
            self.episode_kill_event.wait()
            last_pd_target = None

            # ===== Stage A: snapshot pose, precompute hold tauff =====
            try:
                self.get_robot_data()
            except Exception:
                pass
            _zero_torso_targets()
            if gait_lock:
                self.gait_cycle = np.array([0.25, 0.25])

            hold_pose = np.asarray(self.motorstate, dtype=np.float64).copy()
            try:
                hold_tauff = np.asarray(
                    self.body_ik.compute_whole_body_tau(hold_pose),
                    dtype=np.float64,
                ).copy()
            except Exception as e:
                logger.warning(
                    f"[maintain_standing] hold_tauff compute failed ({e}); using zeros"
                )
                hold_tauff = np.zeros(29, dtype=np.float64)

            # Snapshot hand state so the hand controller holds the
            # operator-placed pose throughout the stand window instead of
            # the default q=0 (Dex3 fully open). At t=stand_window when
            # RTC takes over, apply_action_from_buffer will overwrite
            # hand_shm_array with VLA's hand_cmd. This block only sets
            # the value once per segment; the Dex3 publish process at
            # 100 Hz keeps republishing this hold pose to the motors.
            try:
                hand_hold = np.asarray(self.handstate, dtype=np.float64).copy()
                with self.dual_hand_data_lock:
                    self.hand_shm_array[:] = hand_hold
                logger.info(
                    "[maintain_standing] hand_shm_array seeded with motor snapshot"
                )
            except Exception as e:
                logger.warning(f"[maintain_standing] hand snapshot failed: {e}")

            # ===== Stage B: pure hold + history priming + ff ramp =====
            b_done = 0
            while self.episode_kill_event.is_set() and b_done < hold_ticks:
                try:
                    self.get_robot_data()
                    _zero_torso_targets()
                    if gait_lock:
                        self.gait_cycle = np.array([0.25, 0.25])
                    self.get_ik_observation(record=False)
                    if gait_lock:
                        self.gait_cycle = np.array([0.25, 0.25])
                    self._idx = 0

                    # Full hold_tauff from frame 1. See note above for why
                    # the previous ff ramp was reverted. ff_scale stays as
                    # a tag-embedded constant 1.00 for log compatibility
                    # with the analyzer.
                    ff_scale = 1.0
                    pd_tauff_now = hold_tauff

                    ok, cmd_q = self.safelySetMotor(hold_pose, last_pd_target, pd_tauff_now)
                    if ok:
                        last_pd_target = cmd_q

                    try:
                        tau_est = self.body_ctrl.get_current_motor_tau_est()
                    except Exception:
                        tau_est = None
                    # tag carries the stage and ff_scale so post-hoc CSV
                    # analysis can isolate Stage B ticks and verify the
                    # ff ramp profile without any side-channel logs.
                    WAIST_LOG.log(
                        tag=f"stand_B:ff={ff_scale:.2f}",
                        motorstate=self.motorstate,
                        cmd_q=cmd_q if cmd_q is not None else hold_pose,
                        velstate=self.velstate,
                        tau_est=tau_est,
                        torso_rpy=(self.torso_roll, self.torso_pitch, self.torso_yaw),
                        torso_h=self.torso_height,
                        vxyyaw=(self.vx, self.vy, self.vyaw),
                        have_vla=False,
                        hand_state=getattr(self, "handstate", None),
                        hand_cmd=self.hand_shm_array,
                    )

                    stand_tick += 1
                    b_done += 1
                    # Removed periodic [STAND_B] print: tag is "stand_B:..." in CSV.
                except Exception as e:
                    logger.error(f"[maintain_standing] stage B error: {e}")
                    traceback.print_exc()
                    time.sleep(0.05)
                    continue
                time.sleep(0.02)

            if not self.episode_kill_event.is_set():
                # Killed during Stage B; restart outer loop and re-enter
                # when the event fires again.
                continue

            # ===== Stage C: single AMO call on settled history =====
            pd_target_static = hold_pose.copy()
            pd_tauff_static = hold_tauff.copy()
            try:
                current_lr_arm_q, current_lr_arm_dq = self.get_robot_data()
                _zero_torso_targets()
                if gait_lock:
                    self.gait_cycle = np.array([0.25, 0.25])
                self.get_ik_observation(record=False)
                if gait_lock:
                    self.gait_cycle = np.array([0.25, 0.25])
                self._idx = 0

                _pd_t, _pd_ff, _raw_a = self.body_ik.solve_whole_body_ik(
                    left_wrist=None,
                    right_wrist=None,
                    current_lr_arm_q=current_lr_arm_q,
                    current_lr_arm_dq=current_lr_arm_dq,
                    observation=self.observation,
                    extra_hist=self.extra_hist,
                    is_teleop=False,
                )
                pd_target_static = np.asarray(_pd_t, dtype=np.float64).copy()
                pd_tauff_static = np.asarray(_pd_ff, dtype=np.float64).copy()

                self.last_action = np.concatenate([
                    _raw_a.copy(),
                    (self.motorstate - self.default_dof_pos)[15:] / self.action_scale,
                ])
                logger.info(
                    "[maintain_standing] Stage C: AMO target lower="
                    + np.array2string(pd_target_static[:15], precision=3, suppress_small=True)
                )
                # One-time Stage C marker row in the CSV. cmd_q field
                # carries the AMO-derived static target so it can be
                # extracted offline; motorstate is the live snapshot at
                # the moment of the AMO call.
                try:
                    tau_est_c = self.body_ctrl.get_current_motor_tau_est()
                except Exception:
                    tau_est_c = None
                WAIST_LOG.log(
                    tag="stand_C_amo_target",
                    motorstate=self.motorstate,
                    cmd_q=pd_target_static,
                    velstate=self.velstate,
                    tau_est=tau_est_c,
                    torso_rpy=(self.torso_roll, self.torso_pitch, self.torso_yaw),
                    torso_h=self.torso_height,
                    vxyyaw=(self.vx, self.vy, self.vyaw),
                    have_vla=False,
                    hand_state=getattr(self, "handstate", None),
                    hand_cmd=self.hand_shm_array,
                )
            except Exception as e:
                logger.error(
                    f"[maintain_standing] stage C (single AMO) failed: {e}; "
                    f"falling back to hold-only for this segment"
                )
                traceback.print_exc()
                # pd_target_static and pd_tauff_static fall back to
                # hold_pose / hold_tauff above. Stage D's ramp will be
                # zero-step (already at hold_pose), behaviour reduces to
                # extending Stage B for the rest of the segment.

            # Seed the rate-limiter for the FULL 29-DOF body, not just
            # lower 15. We now ramp arms (15-28) toward AMO's stand-pose
            # arm target as well, instead of pinning them at hold_pose;
            # this is what gives the upper body real PD authority during
            # stand and is the missing piece that earlier left arms feeling
            # soft. Waist (12-14) is still pinned at hold_pose inside the
            # tick loop to avoid the AMO-induced forward bow.
            if last_pd_target is not None:
                body_prev = np.asarray(last_pd_target, dtype=np.float64).copy()
            else:
                body_prev = np.asarray(self.motorstate, dtype=np.float64).copy()

            # ===== Stage D: slow ramp toward pd_target_static, then hold =====
            while self.episode_kill_event.is_set():
                try:
                    self.get_robot_data()
                    _zero_torso_targets()
                    if gait_lock:
                        self.gait_cycle = np.array([0.25, 0.25])
                    self.get_ik_observation(record=False)
                    if gait_lock:
                        self.gait_cycle = np.array([0.25, 0.25])
                    self._idx = 0

                    pd_target = pd_target_static.copy()
                    # === Lower body legs (joints 0-11): ramp to AMO target ===
                    # PD authority builds as cmd diverges from state, so
                    # the legs lift the robot toward AMO's preferred
                    # stand pose actively (not passively).
                    raw_legs = pd_target_static[:12]
                    step_legs = np.clip(
                        raw_legs - body_prev[:12], -lower_rate, lower_rate
                    )
                    target_legs = body_prev[:12] + step_legs
                    legs_smoothed = (
                        lower_alpha * target_legs
                        + (1.0 - lower_alpha) * body_prev[:12]
                    )
                    pd_target[:12] = legs_smoothed
                    # === Waist (joints 12-14): pin at hold_pose ===
                    # AMO's training distribution has waist_p ≈ +0.115 rad
                    # (forward bow). Letting waist follow AMO produces the
                    # visible "弓腰" the user reported earlier, and it is
                    # not necessary for AMO LSTM stability either. Keep
                    # waist at the boot snapshot; the small AMO-vs-hold
                    # delta lands at the RTC handoff and is absorbed by
                    # Fix 2's rate-limit (lower body) and by body_ctrl's
                    # PD for the rest.
                    pd_target[12:15] = hold_pose[12:15]
                    # === Arms (joints 15-28): ramp to AMO target ===
                    # Same rate as legs. Per-tick step is bounded by
                    # lower_rate ≪ π/3, so dynamic_thresholds never
                    # rejects ("ik movement too large" cured). After the
                    # ramp completes the arms are at AMO's default arm
                    # pose with full PD authority — operator-perceived
                    # "softness" of the upper body is gone.
                    raw_arms = pd_target_static[15:]
                    step_arms = np.clip(
                        raw_arms - body_prev[15:], -lower_rate, lower_rate
                    )
                    target_arms = body_prev[15:] + step_arms
                    arms_smoothed = (
                        lower_alpha * target_arms
                        + (1.0 - lower_alpha) * body_prev[15:]
                    )
                    pd_target[15:] = arms_smoothed

                    try:
                        pd_tauff = np.asarray(
                            self.body_ik.compute_whole_body_tau(pd_target),
                            dtype=np.float64,
                        )
                    except Exception as e:
                        logger.warning(f"[maintain_standing] stage D tauff failed: {e}")
                        pd_tauff = pd_tauff_static

                    ok, cmd_q = self.safelySetMotor(pd_target, last_pd_target, pd_tauff)
                    if ok:
                        last_pd_target = cmd_q
                        # Track what was actually sent to motors so the ramp
                        # reference does not drift ahead of reality when
                        # safelySetMotor scales the cmd down. Full 29-DOF
                        # because both legs and arms are now ramping.
                        body_prev = np.asarray(cmd_q, dtype=np.float64).copy()
                    else:
                        continue

                    try:
                        tau_est = self.body_ctrl.get_current_motor_tau_est()
                    except Exception:
                        tau_est = None
                    WAIST_LOG.log(
                        tag="stand_D",
                        motorstate=self.motorstate,
                        cmd_q=cmd_q,
                        velstate=self.velstate,
                        tau_est=tau_est,
                        torso_rpy=(self.torso_roll, self.torso_pitch, self.torso_yaw),
                        torso_h=self.torso_height,
                        vxyyaw=(self.vx, self.vy, self.vyaw),
                        have_vla=False,
                        hand_state=getattr(self, "handstate", None),
                        hand_cmd=self.hand_shm_array,
                    )

                    stand_tick += 1
                    # Removed periodic [STAND_D] print: tag "stand_D" in CSV.
                except Exception as e:
                    logger.error(f"[maintain_standing] stage D error: {e}")
                    traceback.print_exc()
                    time.sleep(0.05)
                    continue
                time.sleep(0.02)

        logger.info("Master: Standing stabilization loop exited.")




    def run_session(self):
        self._session_init()
        last_pd_target = None
        logger.debug("Master: waiting for kill event")
        # self.arm_ctrl.set_weight_to_1()
        self.reset_yaw_offset = False
        self.target_yaw = 0.0  
        self.dyaw = 0.0 
        self.vx = 0.0 
        self.vy = 0.0
        self.vyaw = 0.0
        
        is_first_frame = True
        while not self.episode_kill_event.is_set():
            start_time = time.time()
            logger.debug("Master: looping")
            if is_first_frame:
                self.reset_yaw_offset = True
                is_first_frame = False
            current_lr_arm_q, current_lr_arm_dq = self.get_robot_data()
            motor_time = (
                time.time()
            )  # TODO: might be late here/ consider puting it before getmotorstate


            get_tv_success, head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.get_teleoperator_data()
            )
            # logger.debug("Master: got teleop ddata")

            # self.arm_ctrl.gradually_increase_weight_to_1()
            if not get_tv_success:
                continue
            
            current_h = self.torso_height
            current_rpy = np.array([self.torso_roll, self.torso_pitch, self.torso_yaw], dtype=np.float64)

            # new_h, new_rpy = self.body_ik.solve_lower_ik(
            #     self.motorstate, self.odom_pos, self.quat, left_pose, right_pose, head_rmat, current_h, current_rpy
            # )

            continuous_rot = Rotation.from_euler('xyz', [current_rpy[0], current_rpy[1], current_rpy[2]])
            continuous_quat_xyzw = continuous_rot.as_quat()  # [x, y, z, w]
            continuous_quat_wxyz = np.array([
                continuous_quat_xyzw[3],  # w
                continuous_quat_xyzw[0],  # x
                continuous_quat_xyzw[1],  # y
                continuous_quat_xyzw[2]   # z
            ])

            new_h, new_rpy = self.body_ik.solve_lower_ik(
                self.motorstate, self.odom_pos, continuous_quat_wxyz, 
                left_pose, right_pose, head_rmat, current_h, current_rpy
            )


            self.torso_height = new_h
            self.torso_roll = new_rpy[0]
            self.torso_pitch = new_rpy[1]

            # yaw_diff = new_rpy[2] - current_rpy[2]
            # yaw_diff = np.remainder(yaw_diff + np.pi, 2 * np.pi) - np.pi
            # new_rpy[2] = current_rpy[2] + yaw_diff 

            self.torso_yaw = new_rpy[2]



            self.get_ik_observation()

            pd_target, pd_tauff, raw_action = self.body_ik.solve_whole_body_ik(left_pose, right_pose, current_lr_arm_q, current_lr_arm_dq, self.observation, self.extra_hist)

            self.last_action = np.concatenate([raw_action.copy(), (self.motorstate - self.default_dof_pos)[15:] / self.action_scale])

            vx = self.vx
            vy = self.vy
            vyaw = self.vyaw

            dyaw = self.dyaw

            target_yaw = self.target_yaw



            ik_time = time.time()

            # logger.debug(f"Master: moving motor {sol_q}")
            ok, cmd_q = self.safelySetMotor(pd_target, last_pd_target, pd_tauff)
            if ok:
                last_pd_target = cmd_q
            else:
                continue

            if self.robot == "h1":
                self.setHandMotors(right_qpos, left_qpos)
            elif self.robot == "g1":
                with self.dual_hand_data_lock:
                    self.hand_shm_array[0:7] = left_qpos
                    self.hand_shm_array[7:14] = right_qpos

            # logger.debug("Master: writing data")
            # logger.debug(f"Master: head_rmat: {head_rmat}")
            self.ik_writer.write_data(
                right_qpos,
                left_qpos,
                motor_time,
                ik_time,
                pd_target,
                pd_tauff,
                head_rmat,
                left_pose,
                right_pose,
                new_h,
                new_rpy,
                vx,
                vy,
                vyaw,
                dyaw,
                target_yaw,
            )

            end_time = time.time()

            loop_time = end_time - start_time
            delta_time = CONTROL_DELAY - loop_time
            if delta_time > 0:
                time.sleep(delta_time)
            else:
                print("Loop time takes too much:", loop_time)

            # time.sleep(0.005)
        # self.arm_ctrl.gradually_set_weight_to_0()

    def stop(self):
        self.running = False
        if self.merge_proc is not None and self.merge_proc.is_alive():
            logger.debug("Master: Waiting for merge process to complete...")
            self.merge_proc.join(timeout=10)
            if self.merge_proc.is_alive():
                logger.warning(
                    "Master: Merge process did not complete in time, terminating"
                )
                self.merge_proc.terminate()

        logger.debug("Master: shutting down h1 contorllers...")
        self.body_ctrl.shutdown()
        self.hand_ctrl.shutdown()
        logger.debug("Master: h1 controlleers shutdown")
        logger.info("Master: Stopping all threads ended!")

    def reset(self):
        logger.info("Master: Resetting RobotTaskmaster...")
        if self.running:
            self.stop()
        logger.info("Master: Clearing stop event...")
        # self.kill_event.clear()  # TODO: create a new one?

        self.hand_ctrl.reset()
        self.body_ctrl.reset()
        self.first = True
        self.running = False

        self.robot_shm_array[:] = 0

        self.ik_writer = IKDataWriter(self.shared_data["dirname"])

        # if hasattr(self, 'tau_file'):
        #     self.tau_file.close()
        #     print(f"Tau data saved to {self.tau_log_path}")

        logger.info("RobotTaskmaster has been reset and is ready to start again.")

    def merge_data(self):
        if self.ik_writer is not None:
            self.ik_writer.close()

        if self.merge_proc is not None and self.merge_proc.is_alive():
            logger.debug(
                "Master: Previous merge process still running, not starting a new one"
            )
            return

        def merge_process():
            merger = DataMerger(self.shared_data["dirname"])
            merger.merge_json()

        self.merge_proc = Process(target=merge_process)
        self.merge_proc.daemon = True
        self.merge_proc.start()
        logger.debug("Master: Started merge process in background")

    def delete_last_data(self):
        # TODO: auto delete
        with open(self.shared_data["dirname"] + "/failed", "w"):
            pass

    def ctrl_whole_body(self, pred_action): # TODO: this is just a simple refactor. need to later refactor this out to RoboControllers
        """
        pred_action: np.array of shape (32,)
        """
        arm_poseList = pred_action[:14]
        hand_poseList = pred_action[14:28]
        current_lr_arm_q, current_lr_arm_dq = self.get_robot_data()
        self.torso_roll = pred_action[28]
        self.torso_pitch = pred_action[29]
        self.torso_yaw = pred_action[30]
        self.torso_height = pred_action[31]

        print("predicted torso r, p, y, h:", pred_action[28], pred_action[29], pred_action[30], pred_action[31])
        self.get_ik_observation()
        pd_target, pd_tauff, raw_action = self.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=self.observation,
            extra_hist=self.extra_hist,
            is_teleop=False
        )
        self.last_action = np.concatenate([raw_action.copy(), (self.motorstate - self.default_dof_pos)[15:] / self.action_scale])
        pd_target[15:] = arm_poseList
        pd_tauff[15:] = self.get_tauer(np.array(arm_poseList))

        with self.dual_hand_data_lock:
            self.hand_shm_array[:] = hand_poseList

        self.body_ctrl.ctrl_whole_body(pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15])

