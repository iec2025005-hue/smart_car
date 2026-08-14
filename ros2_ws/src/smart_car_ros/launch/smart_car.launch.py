import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='smart_car_ros',
            executable='vision_node',
            name='vision_node',
            output='screen'
        ),
        Node(
            package='smart_car_ros',
            executable='navigation_node',
            name='navigation_node',
            output='screen'
        ),
        Node(
            package='smart_car_ros',
            executable='esp32_bridge_node',
            name='esp32_bridge_node',
            output='screen'
        )
    ])
