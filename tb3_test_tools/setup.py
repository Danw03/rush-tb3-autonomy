import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'tb3_test_tools'


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join(
                'share',
                package_name,
                'data',
                'camera',
            ),
            glob('data/camera/*'),
        ),
        (
            os.path.join(
                'share',
                package_name,
                'data',
                'lidar',
            ),
            glob('data/lidar/*'),
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch',
            ),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rush',
    maintainer_email='rush@example.com',
    description=(
        'Offline sample data publishers for '
        'TurtleBot3 autonomy node testing.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            (
                'image_sample_publisher = '
                'tb3_test_tools.image_sample_publisher:main'
            ),
            (
                'video_sample_publisher = '
                'tb3_test_tools.video_sample_publisher:main'
            ),
            (
                'scan_sample_publisher = '
                'tb3_test_tools.scan_sample_publisher:main'
            ),
        ],
    },
)