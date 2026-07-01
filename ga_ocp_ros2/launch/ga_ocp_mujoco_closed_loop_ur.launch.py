from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    ros2_share = get_package_share_directory('ga_ocp_ros2')
    config = f"{ros2_share}/config/closed_loop_mpc_ur.yaml"

    backend = LaunchConfiguration('backend')
    solve_budget_ms = LaunchConfiguration('solve_budget_ms')
    acceleration_weight = LaunchConfiguration('acceleration_weight')
    enforce_solve_budget = LaunchConfiguration('enforce_solve_budget')
    duration_s = LaunchConfiguration('duration_s')
    dt = LaunchConfiguration('dt')
    horizon = LaunchConfiguration('horizon')
    control_rate_hz = LaunchConfiguration('control_rate_hz')
    mass_scale = LaunchConfiguration('mass_scale')
    plant_payload_mass = LaunchConfiguration('plant_payload_mass')
    controller_payload_mass = LaunchConfiguration('controller_payload_mass')
    model_payload = LaunchConfiguration('model_payload')
    output_prefix = LaunchConfiguration('output_prefix')
    enable_collision_cost = LaunchConfiguration('enable_collision_cost')
    collision_obstacle_count = LaunchConfiguration('collision_obstacle_count')
    collision_weight = LaunchConfiguration('collision_weight')
    collision_safety_distance = LaunchConfiguration('collision_safety_distance')
    enable_viewer = LaunchConfiguration('enable_viewer')
    payload_body_name = LaunchConfiguration('payload_body_name')
    external_force_body_name = LaunchConfiguration('external_force_body_name')
    external_force_start_s = LaunchConfiguration('external_force_start_s')
    external_force_duration_s = LaunchConfiguration('external_force_duration_s')
    link_com_offset_x = LaunchConfiguration('link_com_offset_x')
    link_com_offset_y = LaunchConfiguration('link_com_offset_y')
    link_com_offset_z = LaunchConfiguration('link_com_offset_z')

    plant_payload_com_x = LaunchConfiguration('plant_payload_com_x')
    plant_payload_com_y = LaunchConfiguration('plant_payload_com_y')
    plant_payload_com_z = LaunchConfiguration('plant_payload_com_z')
    controller_payload_com_x = LaunchConfiguration('controller_payload_com_x')
    controller_payload_com_y = LaunchConfiguration('controller_payload_com_y')
    controller_payload_com_z = LaunchConfiguration('controller_payload_com_z')
    external_force_x = LaunchConfiguration('external_force_x')
    external_force_y = LaunchConfiguration('external_force_y')
    external_force_z = LaunchConfiguration('external_force_z')
    external_torque_x = LaunchConfiguration('external_torque_x')
    external_torque_y = LaunchConfiguration('external_torque_y')
    external_torque_z = LaunchConfiguration('external_torque_z')

    closed_loop_node = Node(
        package='ga_ocp_ros2',
        executable='closed_loop_mpc_node',
        name='closed_loop_mpc_node',
        output='screen',
        parameters=[
            config,
            {
                'backend': backend,
                'solve_budget_ms': ParameterValue(solve_budget_ms, value_type=float),
                'acceleration_weight': ParameterValue(acceleration_weight, value_type=float),
                'enforce_solve_budget': ParameterValue(enforce_solve_budget, value_type=bool),
                'experiment_duration_s': ParameterValue(duration_s, value_type=float),
                'dt': ParameterValue(dt, value_type=float),
                'horizon': ParameterValue(horizon, value_type=int),
                'control_rate_hz': ParameterValue(control_rate_hz, value_type=float),
                'plant_mass_scale': ParameterValue(mass_scale, value_type=float),
                'plant_payload_mass': ParameterValue(plant_payload_mass, value_type=float),
                'controller_payload_mass': ParameterValue(controller_payload_mass, value_type=float),
                'model_payload': ParameterValue(model_payload, value_type=bool),
                'plant_payload_com_x': ParameterValue(plant_payload_com_x, value_type=float),
                'plant_payload_com_y': ParameterValue(plant_payload_com_y, value_type=float),
                'plant_payload_com_z': ParameterValue(plant_payload_com_z, value_type=float),
                'payload_com_attachment_x': ParameterValue(controller_payload_com_x, value_type=float),
                'payload_com_attachment_y': ParameterValue(controller_payload_com_y, value_type=float),
                'payload_com_attachment_z': ParameterValue(controller_payload_com_z, value_type=float),
                'external_force_body_name': external_force_body_name,
                'external_force_start_s': ParameterValue(external_force_start_s, value_type=float),
                'external_force_duration_s': ParameterValue(external_force_duration_s, value_type=float),
                'external_force_x': ParameterValue(external_force_x, value_type=float),
                'external_force_y': ParameterValue(external_force_y, value_type=float),
                'external_force_z': ParameterValue(external_force_z, value_type=float),
                'external_torque_x': ParameterValue(external_torque_x, value_type=float),
                'external_torque_y': ParameterValue(external_torque_y, value_type=float),
                'external_torque_z': ParameterValue(external_torque_z, value_type=float),
                'output_prefix': output_prefix,
                'enable_collision_cost': ParameterValue(enable_collision_cost, value_type=bool),
                'collision_obstacle_count': ParameterValue(collision_obstacle_count, value_type=int),
                'collision_weight': ParameterValue(collision_weight, value_type=float),
                'collision_safety_distance': ParameterValue(collision_safety_distance, value_type=float),
            },
        ],
    )

    mujoco_executor_node = Node(
        package='ga_ocp_ros2',
        executable='joint_command_executor.py',
        name='mujoco_joint_executor_node',
        output='screen',
        parameters=[
            {
                'robot': 'ur',
                'mass_scale': ParameterValue(mass_scale, value_type=float),
                'payload_mass': ParameterValue(plant_payload_mass, value_type=float),
                'payload_body_name': payload_body_name,
                'payload_com_x': ParameterValue(plant_payload_com_x, value_type=float),
                'payload_com_y': ParameterValue(plant_payload_com_y, value_type=float),
                'payload_com_z': ParameterValue(plant_payload_com_z, value_type=float),
                'link_com_offset_x': ParameterValue(link_com_offset_x, value_type=float),
                'link_com_offset_y': ParameterValue(link_com_offset_y, value_type=float),
                'link_com_offset_z': ParameterValue(link_com_offset_z, value_type=float),
                'enable_viewer': ParameterValue(enable_viewer, value_type=bool),
                'external_force_body_name': external_force_body_name,
                'external_force_start_s': ParameterValue(external_force_start_s, value_type=float),
                'external_force_duration_s': ParameterValue(external_force_duration_s, value_type=float),
                'external_force_x': ParameterValue(external_force_x, value_type=float),
                'external_force_y': ParameterValue(external_force_y, value_type=float),
                'external_force_z': ParameterValue(external_force_z, value_type=float),
                'external_torque_x': ParameterValue(external_torque_x, value_type=float),
                'external_torque_y': ParameterValue(external_torque_y, value_type=float),
                'external_torque_z': ParameterValue(external_torque_z, value_type=float),
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='tetrapga'),
        DeclareLaunchArgument('acceleration_weight', default_value='0.0'),
        DeclareLaunchArgument('solve_budget_ms', default_value='10.0'),
        DeclareLaunchArgument('enforce_solve_budget', default_value='true'),
        DeclareLaunchArgument('duration_s', default_value='20.0'),
        DeclareLaunchArgument('dt', default_value='0.008'),
        DeclareLaunchArgument('horizon', default_value='40'),
        DeclareLaunchArgument('control_rate_hz', default_value='125.0'),
        DeclareLaunchArgument('mass_scale', default_value='1.0'),
        DeclareLaunchArgument('plant_payload_mass', default_value='0.0'),
        DeclareLaunchArgument('controller_payload_mass', default_value='0.0'),
        DeclareLaunchArgument('model_payload', default_value='false'),
        DeclareLaunchArgument('plant_payload_com_x', default_value='0.0'),
        DeclareLaunchArgument('plant_payload_com_y', default_value='0.0'),
        DeclareLaunchArgument('plant_payload_com_z', default_value='0.05'),
        DeclareLaunchArgument('controller_payload_com_x', default_value='0.0'),
        DeclareLaunchArgument('controller_payload_com_y', default_value='0.0'),
        DeclareLaunchArgument('controller_payload_com_z', default_value='0.05'),
        DeclareLaunchArgument('payload_body_name', default_value='attachment'),
        DeclareLaunchArgument('link_com_offset_x', default_value='0.0'),
        DeclareLaunchArgument('link_com_offset_y', default_value='0.0'),
        DeclareLaunchArgument('link_com_offset_z', default_value='0.0'),
        DeclareLaunchArgument('enable_viewer', default_value='true'),
        DeclareLaunchArgument('external_force_body_name', default_value='wrist_3_link'),
        DeclareLaunchArgument('external_force_start_s', default_value='-1.0'),
        DeclareLaunchArgument('external_force_duration_s', default_value='0.0'),
        DeclareLaunchArgument('external_force_x', default_value='0.0'),
        DeclareLaunchArgument('external_force_y', default_value='0.0'),
        DeclareLaunchArgument('external_force_z', default_value='0.0'),
        DeclareLaunchArgument('external_torque_x', default_value='0.0'),
        DeclareLaunchArgument('external_torque_y', default_value='0.0'),
        DeclareLaunchArgument('external_torque_z', default_value='0.0'),
        DeclareLaunchArgument('output_prefix', default_value=''),
        DeclareLaunchArgument('enable_collision_cost', default_value='false'),
        DeclareLaunchArgument('collision_obstacle_count', default_value='4'),
        DeclareLaunchArgument('collision_weight', default_value='50.0'),
        DeclareLaunchArgument('collision_safety_distance', default_value='0.08'),
        closed_loop_node,
        mujoco_executor_node,
    ])
