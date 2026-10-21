"""Training-matched three-camera scene for ACT; controllers wait for act_rollout."""
import atexit
import math
from pathlib import Path
import shutil
import tempfile
from xml.etree import ElementTree as ET

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from robot_bringup.configuration import prepare_configuration
from robot_bringup.profile import load_profile, package_file


def setup(context):
    get = lambda key: LaunchConfiguration(key).perform(context)
    x, y = float(get('bottle_x')), float(get('bottle_y'))
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError('Bottle coordinates must be finite')
    runtime = Path(tempfile.mkdtemp(prefix='act_gazebo_'))
    try:
        profile = load_profile('kinova_gen3_lite')
        _, _, _, urdf, controllers = prepare_configuration(profile, runtime, 'known', True,
                                                           demonstration_cameras=True)
        simulation = profile['simulation']
        world = ET.parse(package_file(simulation['package'], 'worlds/' + simulation['world_file'], 'ACT world'))
        pose = world.getroot().find("world/model[@name='plastic_bottle']/pose")
        if pose is None:
            raise ValueError('Training scene is missing plastic_bottle')
        values = pose.text.split()
        values[:2] = [str(x), str(y)]
        pose.text = ' '.join(values)
        world_file = runtime / 'world.sdf'
        world.write(world_file, encoding='unicode', xml_declaration=True)
        gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(package_file(
            simulation['package'], 'launch/' + simulation['launch_file'], 'ACT simulation launch'))),
            launch_arguments={'robot_description_file': str(urdf), 'controllers_file': str(controllers),
                              'world_file': str(world_file), 'use_camera': 'true', 'record_images': 'true',
                              'headless': get('headless'), 'use_sim_time': 'true'}.items())
        atexit.register(shutil.rmtree, runtime, ignore_errors=True)
        return [LogInfo(msg=f'ACT Gazebo: bottle=({x:.4f}, {y:.4f}); three RGB views; runtime={runtime}'),
                LogInfo(msg='Run python -m robot_bringup.act_rollout --checkpoint <best.pt> in your PyTorch venv.'),
                gazebo]
    except BaseException:
        shutil.rmtree(runtime, ignore_errors=True)
        raise


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bottle_x', default_value='0.35'),
        DeclareLaunchArgument('bottle_y', default_value='0.15'),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false']),
        OpaqueFunction(function=setup),
    ])
