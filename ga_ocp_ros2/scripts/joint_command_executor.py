#!/usr/bin/env python3
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory


@dataclass(frozen=True)
class RobotConfig:
    robot: str
    scene_relative_path: str
    joint_state_names: List[str]
    mujoco_joint_names: List[str]
    actuator_names: List[str]
    default_target: np.ndarray
    control_mode: str
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray


def _array(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


_PARAM_SENTINEL = 1.0e100


def _vector3_from_params(
    node: Node,
    vector_name: str,
    scalar_prefix: str,
    default: List[float],
) -> np.ndarray:
    value = np.asarray(node.declare_parameter(vector_name, default).value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f'{vector_name} must contain exactly 3 values.')

    scalar_values = [
        float(node.declare_parameter(f'{scalar_prefix}_{axis}', _PARAM_SENTINEL).value)
        for axis in ('x', 'y', 'z')
    ]
    scalar_set = [abs(v) < 0.5 * _PARAM_SENTINEL for v in scalar_values]
    if any(scalar_set):
        if not all(scalar_set):
            raise ValueError(f'{scalar_prefix}_x/y/z must be set together.')
        value = np.asarray(scalar_values, dtype=float)
    return value


ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    'ur': RobotConfig(
        robot='ur',
        scene_relative_path='robot-assets/ur10/mjcf/scene.xml',
        joint_state_names=[
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ],
        mujoco_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        actuator_names=['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6'],
        default_target=_array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]),
        control_mode='direct',
        kp=np.zeros(6, dtype=float),
        kd=np.zeros(6, dtype=float),
        effort_limit=np.full(6, np.inf, dtype=float),
    ),
    'franka': RobotConfig(
        robot='franka',
        scene_relative_path='robot-assets/franka_panda/mjcf/scene.xml',
        joint_state_names=[
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ],
        mujoco_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
        actuator_names=['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'],
        default_target=np.zeros(7, dtype=float),
        control_mode='impedance_torque',
        kp=_array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0]),
        kd=_array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0]),
        effort_limit=_array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
    ),
    'leap_left': RobotConfig(
        robot='leap_left',
        scene_relative_path='robot-assets/leap_hand/mjcf/scene_left.xml',
        joint_state_names=[str(i) for i in range(16)],
        mujoco_joint_names=[
            'if_mcp', 'if_rot', 'if_pip', 'if_dip',
            'mf_mcp', 'mf_rot', 'mf_pip', 'mf_dip',
            'rf_mcp', 'rf_rot', 'rf_pip', 'rf_dip',
            'th_cmc', 'th_axl', 'th_mcp', 'th_ipl',
        ],
        actuator_names=[
            'if_mcp_act', 'if_rot_act', 'if_pip_act', 'if_dip_act',
            'mf_mcp_act', 'mf_rot_act', 'mf_pip_act', 'mf_dip_act',
            'rf_mcp_act', 'rf_rot_act', 'rf_pip_act', 'rf_dip_act',
            'th_cmc_act', 'th_axl_act', 'th_mcp_act', 'th_ipl_act',
        ],
        default_target=np.zeros(16, dtype=float),
        control_mode='direct',
        kp=np.zeros(16, dtype=float),
        kd=np.zeros(16, dtype=float),
        effort_limit=np.full(16, np.inf, dtype=float),
    ),
    'stanford_tidybot': RobotConfig(
        robot='stanford_tidybot',
        scene_relative_path='robot-assets/stanford_tidybot/mjcf/scene.xml',
        joint_state_names=[
            'base_x_joint',
            'base_y_joint',
            'base_yaw_joint',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7',
        ],
        mujoco_joint_names=[
            'joint_x',
            'joint_y',
            'joint_th',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7',
        ],
        actuator_names=[
            'joint_x',
            'joint_y',
            'joint_th',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7',
        ],
        default_target=_array([
            0.0, 0.0, 0.0,
            0.0, 0.26179939, 3.14159265, -2.26892803,
            0.0, 0.95993109, 1.57079633,
        ]),
        control_mode='direct',
        kp=np.zeros(10, dtype=float),
        kd=np.zeros(10, dtype=float),
        effort_limit=_array([1000.0, 1000.0, 1000.0, 39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]),
    ),
    'tidybot': RobotConfig(
        robot='stanford_tidybot',
        scene_relative_path='robot-assets/stanford_tidybot/mjcf/scene.xml',
        joint_state_names=[
            'base_x_joint',
            'base_y_joint',
            'base_yaw_joint',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7',
        ],
        mujoco_joint_names=[
            'joint_x',
            'joint_y',
            'joint_th',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7',
        ],
        actuator_names=[
            'joint_x',
            'joint_y',
            'joint_th',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7',
        ],
        default_target=_array([
            0.0, 0.0, 0.0,
            0.0, 0.26179939, 3.14159265, -2.26892803,
            0.0, 0.95993109, 1.57079633,
        ]),
        control_mode='direct',
        kp=np.zeros(10, dtype=float),
        kd=np.zeros(10, dtype=float),
        effort_limit=_array([1000.0, 1000.0, 1000.0, 39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]),
    ),
}


class MujocoJointExecutor(Node):
    def __init__(self) -> None:
        super().__init__('mujoco_joint_executor_node')

        self.robot = str(self.declare_parameter('robot', 'ur').value)
        if self.robot not in ROBOT_CONFIGS:
            raise ValueError(f"Unsupported robot='{self.robot}'. Expected one of {sorted(ROBOT_CONFIGS.keys())}.")
        self.config = ROBOT_CONFIGS[self.robot]

        self.command_topic = self.declare_parameter('command_topic', '/joint_commands').value
        self.joint_state_topic = self.declare_parameter('joint_state_topic', '/joint_states').value
        self.position_tolerance = float(self.declare_parameter('position_tolerance', 1e-2).value)
        self.xml_file_param = str(self.declare_parameter('xml_file', '').value)
        self.mass_scale = float(self.declare_parameter('mass_scale', 1.0).value)
        self.payload_mass = float(self.declare_parameter('payload_mass', 0.0).value)
        self.payload_body_name = str(self.declare_parameter('payload_body_name', 'attachment').value)
        self.payload_com = _vector3_from_params(
            self, 'payload_com', 'payload_com', [0.0, 0.0, 0.05]
        )
        self.link_com_offset = _vector3_from_params(
            self, 'link_com_offset', 'link_com_offset', [0.0, 0.0, 0.0]
        )
        self.enable_viewer = bool(self.declare_parameter('enable_viewer', True).value)
        self.external_force_body_name = str(
            self.declare_parameter('external_force_body_name', 'wrist_3_link').value
        )
        self.external_force_start_s = float(
            self.declare_parameter('external_force_start_s', -1.0).value
        )
        self.external_force_duration_s = float(
            self.declare_parameter('external_force_duration_s', 0.0).value
        )
        self.external_force = _vector3_from_params(
            self, 'external_force', 'external_force', [0.0, 0.0, 0.0]
        )
        self.external_torque = _vector3_from_params(
            self, 'external_torque', 'external_torque', [0.0, 0.0, 0.0]
        )

        core_share = get_package_share_directory('ga_ocp_core')
        self.xml_file = self.xml_file_param or f"{core_share}/{self.config.scene_relative_path}"

        self.joint_names = list(self.config.joint_state_names)
        self.mujoco_joint_names = list(self.config.mujoco_joint_names)
        self.actuator_names = list(self.config.actuator_names)
        self.n = len(self.joint_names)

        self.publish_joint_state = self.create_publisher(JointState, self.joint_state_topic, 20)
        self.create_subscription(JointTrajectory, self.command_topic, self.trajectory_callback, 10)
        self.create_subscription(String, '/planning_status', self.status_callback, 10)

        self.trajectory: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self.trajectory_start_time: Optional[float] = None
        self.default_target = self.config.default_target.copy()
        self.position_target = self.default_target.copy()
        self.velocity_target = np.zeros(self.n, dtype=float)
        self.effort_target = np.zeros(self.n, dtype=float)

        self.stop_requested = False
        self.paused = False
        self.qpos_addrs: List[int] = []
        self.qvel_addrs: List[int] = []
        self.ctrl_addrs: List[int] = []
        self.external_force_body_id: Optional[int] = None

        self.get_logger().info(
            f"MuJoCo executor ready. robot={self.robot}, mode={self.config.control_mode}, "
            f"model={self.xml_file}, cmd={self.command_topic}, state={self.joint_state_topic}, "
            f"mass_scale={self.mass_scale:.3f}, payload_mass={self.payload_mass:.3f}, "
            f"payload_com={self.payload_com.tolist()}, "
            f"link_com_offset={self.link_com_offset.tolist()}, viewer={self.enable_viewer}, "
            f"external_body={self.external_force_body_name}, "
            f"external_start={self.external_force_start_s:.3f}, "
            f"external_duration={self.external_force_duration_s:.3f}, "
            f"external_force={self.external_force.tolist()}, "
            f"external_torque={self.external_torque.tolist()}"
        )

    def _lookup_body_id(self, model: mujoco.MjModel, body_name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in {self.xml_file}")
        return int(body_id)

    def _apply_runtime_model_variations(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dirty = False
        ur_moving_body_names = [
            'shoulder_link',
            'upper_arm_link',
            'forearm_link',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
        ]

        if self.robot == 'ur' and abs(self.mass_scale - 1.0) > 1e-9:
            for body_name in ur_moving_body_names:
                body_id = self._lookup_body_id(model, body_name)
                model.body_mass[body_id] *= self.mass_scale
                model.body_inertia[body_id] *= self.mass_scale
            dirty = True
            self.get_logger().info(f'Applied UR10 mass/inertia scale {self.mass_scale:.3f} to MuJoCo plant.')

        if self.robot == 'ur' and np.linalg.norm(self.link_com_offset) > 0.0:
            for body_name in ur_moving_body_names:
                body_id = self._lookup_body_id(model, body_name)
                model.body_ipos[body_id] = model.body_ipos[body_id] + self.link_com_offset
            dirty = True
            self.get_logger().info(
                f'Applied UR10 six-link COM offset {self.link_com_offset.tolist()} to MuJoCo plant.'
            )

        if self.payload_mass > 0.0:
            body_id = self._lookup_body_id(model, self.payload_body_name)
            model.body_mass[body_id] = self.payload_mass
            model.body_inertia[body_id] = np.full(3, max(1e-6, 1e-6 * self.payload_mass), dtype=float)
            model.body_ipos[body_id] = self.payload_com
            dirty = True
            self.get_logger().info(
                f"Attached payload mass={self.payload_mass:.3f}kg com={self.payload_com.tolist()} "
                f"to body '{self.payload_body_name}'."
            )

        if dirty:
            mujoco.mj_setConst(model, data)

    def _lookup_external_force_body(self, model: mujoco.MjModel) -> None:
        has_force = np.linalg.norm(self.external_force) > 0.0
        has_torque = np.linalg.norm(self.external_torque) > 0.0
        if self.external_force_duration_s <= 0.0 or not (has_force or has_torque):
            self.external_force_body_id = None
            return
        self.external_force_body_id = self._lookup_body_id(model, self.external_force_body_name)
        self.get_logger().info(
            f"External wrench enabled on body '{self.external_force_body_name}' "
            f"from t={self.external_force_start_s:.3f}s for "
            f"{self.external_force_duration_s:.3f}s."
        )

    def _apply_external_wrench(self, data: mujoco.MjData) -> None:
        data.xfrc_applied[:, :] = 0.0
        if self.external_force_body_id is None:
            return
        if data.time < self.external_force_start_s:
            return
        if data.time >= self.external_force_start_s + self.external_force_duration_s:
            return
        # MuJoCo spatial wrench order is torque first, then force.
        data.xfrc_applied[self.external_force_body_id, 0:3] = self.external_torque
        data.xfrc_applied[self.external_force_body_id, 3:6] = self.external_force

    def _lookup_joint_addresses(self, model: mujoco.MjModel) -> None:
        self.qpos_addrs = []
        self.qvel_addrs = []
        for name in self.mujoco_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in {self.xml_file}")
            self.qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
            self.qvel_addrs.append(int(model.jnt_dofadr[joint_id]))

        self.ctrl_addrs = []
        for name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in {self.xml_file}")
            self.ctrl_addrs.append(int(actuator_id))

    def _joint_vector_from_data(self, data: mujoco.MjData, addrs: List[int], source: np.ndarray) -> np.ndarray:
        return np.asarray([float(source[idx]) for idx in addrs], dtype=float)

    def _build_command_from_msg(
        self, msg: JointTrajectory, point_index: int
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        point = msg.points[point_index]
        if msg.joint_names:
            name_to_idx = {name: idx for idx, name in enumerate(msg.joint_names)}
            if not all(name in name_to_idx for name in self.joint_names):
                return None
            if any(name_to_idx[name] >= len(point.positions) for name in self.joint_names):
                return None
            positions = np.asarray([point.positions[name_to_idx[name]] for name in self.joint_names], dtype=float)
            if len(point.velocities) >= len(msg.joint_names):
                velocities = np.asarray([point.velocities[name_to_idx[name]] for name in self.joint_names], dtype=float)
            else:
                velocities = np.zeros(self.n, dtype=float)
            if len(point.effort) >= len(msg.joint_names):
                efforts = np.asarray([point.effort[name_to_idx[name]] for name in self.joint_names], dtype=float)
            else:
                efforts = np.zeros(self.n, dtype=float)
        else:
            if len(point.positions) < self.n:
                return None
            positions = np.asarray(point.positions[:self.n], dtype=float)
            velocities = np.asarray(point.velocities[:self.n], dtype=float) if len(point.velocities) >= self.n else np.zeros(self.n, dtype=float)
            efforts = np.asarray(point.effort[:self.n], dtype=float) if len(point.effort) >= self.n else np.zeros(self.n, dtype=float)

        time_from_start = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        return time_from_start, positions, velocities, efforts

    def status_callback(self, msg: String) -> None:
        if 'finished' in msg.data:
            self.stop_requested = True

    def trajectory_callback(self, msg: JointTrajectory) -> None:
        trajectory: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for point_index in range(len(msg.points)):
            command = self._build_command_from_msg(msg, point_index)
            if command is None:
                self.get_logger().warning('Discarded joint trajectory with incompatible joint layout.')
                return
            trajectory.append(command)

        if not trajectory:
            return

        self.trajectory = trajectory
        self.trajectory_start_time = time.time()
        self.position_target = trajectory[0][1].copy()
        self.velocity_target = trajectory[0][2].copy()
        self.effort_target = trajectory[0][3].copy()

    def _sample_active_command(self, now: float) -> None:
        if not self.trajectory or self.trajectory_start_time is None:
            return

        elapsed = max(0.0, now - self.trajectory_start_time)
        active = self.trajectory[-1]
        for command in self.trajectory:
            if elapsed <= command[0]:
                active = command
                break

        self.position_target = active[1].copy()
        self.velocity_target = active[2].copy()
        self.effort_target = active[3].copy()

        if elapsed > self.trajectory[-1][0] + 1.0:
            self.trajectory = []
            self.trajectory_start_time = None

    def key_callback(self, keycode: int) -> None:
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def _publish_joint_state(self, data: mujoco.MjData) -> np.ndarray:
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = list(self.joint_names)
        joint_state_msg.position = self._joint_vector_from_data(data, self.qpos_addrs, data.qpos).tolist()
        joint_state_msg.velocity = self._joint_vector_from_data(data, self.qvel_addrs, data.qvel).tolist()
        joint_state_msg.effort = self._joint_vector_from_data(data, self.ctrl_addrs, data.actuator_force).tolist()
        self.publish_joint_state.publish(joint_state_msg)
        return np.asarray(joint_state_msg.position, dtype=float)

    def _apply_control(self, data: mujoco.MjData) -> None:
        if self.config.control_mode == 'direct':
            data.ctrl[self.ctrl_addrs] = self.position_target
            return

        if self.config.control_mode == 'impedance_torque':
            q = self._joint_vector_from_data(data, self.qpos_addrs, data.qpos)
            dq = self._joint_vector_from_data(data, self.qvel_addrs, data.qvel)
            torque = self.config.kp * (self.position_target - q) + self.config.kd * (self.velocity_target - dq)
            torque += self.effort_target
            torque = np.clip(torque, -self.config.effort_limit, self.config.effort_limit)
            data.ctrl[self.ctrl_addrs] = torque
            return

        raise ValueError(f"Unsupported control mode '{self.config.control_mode}'")

    def _run_loop(self, model: mujoco.MjModel, data: mujoco.MjData, viewer=None) -> None:
        while rclpy.ok() and not self.stop_requested and (viewer is None or viewer.is_running()):
            step_start = time.time()

            self._sample_active_command(step_start)
            current_position = self._publish_joint_state(data)
            position_error = np.max(np.abs(current_position - self.position_target))

            if not self.paused:
                self._apply_control(data)
                self._apply_external_wrench(data)
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()

            if position_error < self.position_tolerance and not self.trajectory:
                self.position_target = current_position.copy()

            rclpy.spin_once(self, timeout_sec=0.0)

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def run(self) -> None:
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        self._apply_runtime_model_variations(model, data)
        self._lookup_external_force_body(model)
        self._lookup_joint_addresses(model)
        data.qpos[self.qpos_addrs] = self.default_target
        data.qvel[self.qvel_addrs] = 0.0
        data.ctrl[self.ctrl_addrs] = self.default_target
        mujoco.mj_forward(model, data)

        if self.enable_viewer:
            with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
                self._run_loop(model, data, viewer)
        else:
            self._run_loop(model, data, None)


def main() -> None:
    rclpy.init()
    node = MujocoJointExecutor()

    def request_stop(signum, frame):
        del signum, frame
        node.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
