from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    ros2_share = get_package_share_directory('ga_ocp_ros2')
    config = f"{ros2_share}/config/closed_loop_mpc_tidybot.yaml"

    backend = LaunchConfiguration('backend')
    solve_budget_ms = LaunchConfiguration('solve_budget_ms')
    acceleration_weight = LaunchConfiguration('acceleration_weight')
    enforce_solve_budget = LaunchConfiguration('enforce_solve_budget')
    duration_s = LaunchConfiguration('duration_s')
    dt = LaunchConfiguration('dt')
    horizon = LaunchConfiguration('horizon')
    control_rate_hz = LaunchConfiguration('control_rate_hz')
    output_prefix = LaunchConfiguration('output_prefix')
    enable_viewer = LaunchConfiguration('enable_viewer')

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
                'output_prefix': output_prefix,
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
                'robot': 'stanford_tidybot',
                'enable_viewer': ParameterValue(enable_viewer, value_type=bool),
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='tetrapga'),
        DeclareLaunchArgument('acceleration_weight', default_value='0.0'),
        DeclareLaunchArgument('solve_budget_ms', default_value='10.0'),
        DeclareLaunchArgument('enforce_solve_budget', default_value='true'),
        DeclareLaunchArgument('duration_s', default_value='20.0'),
        DeclareLaunchArgument('dt', default_value='0.02'),
        DeclareLaunchArgument('horizon', default_value='20'),
        DeclareLaunchArgument('control_rate_hz', default_value='50.0'),
        DeclareLaunchArgument('enable_viewer', default_value='true'),
        DeclareLaunchArgument('output_prefix', default_value=''),
        RegisterEventHandler(
            OnProcessExit(
                target_action=closed_loop_node,
                on_exit=[EmitEvent(event=Shutdown(reason='closed-loop mpc node finished'))],
            )
        ),
        closed_loop_node,
        mujoco_executor_node,
    ])
