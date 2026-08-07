"""Setuptools configuration for the tb3_reference ROS 2 package."""

from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'tb3_reference'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rush',
    maintainer_email='rush@todo.todo',
    description='Generate an odom-frame MPC reference from the cone path.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_reference_node = '
            'tb3_reference.camera_reference_node:main',
            'reference_generator_node = '
            'tb3_reference.reference_generator_node:main',
        ],
    },
)
