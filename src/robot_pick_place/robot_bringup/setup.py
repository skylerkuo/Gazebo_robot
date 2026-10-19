from glob import glob
from setuptools import setup

setup(
    name='robot_bringup', version='0.1.0', packages=['robot_bringup'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/robot_bringup']),
        ('share/robot_bringup', ['package.xml', 'LICENSE']),
        ('share/robot_bringup/launch', glob('launch/*.py')),
        ('share/robot_bringup/config/robots', glob('config/robots/*.yaml')),
        ('share/robot_bringup/config/scenes', glob('config/scenes/*.yaml')),
    ],
    install_requires=['setuptools', 'PyYAML'], zip_safe=True,
    maintainer='Skyler Kuo', maintainer_email='skylerkuo@users.noreply.github.com',
    description='Profile-based integration of robot simulation, MoveIt and generic tasks.',
    license='BSD-3-Clause',
    entry_points={'console_scripts': ['wait_ready = robot_bringup.readiness:main',
                                     'pick_console = robot_bringup.pick_console:main']},
)
