from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'smart_car_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'pyserial', 'ultralytics', 'opencv-python'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS 2 package for Smart Car with Swerve Drive and YOLO vision',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = smart_car_ros.vision_node:main',
            'navigation_node = smart_car_ros.navigation_node:main',
            'esp32_bridge_node = smart_car_ros.esp32_bridge_node:main',
        ],
    },
)
