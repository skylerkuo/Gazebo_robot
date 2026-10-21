"""Build one shared MoveIt parameter dictionary from explicit profile resources."""
from pathlib import Path
from xml.etree import ElementTree as ET

import xacro
import yaml

from .profile import ProfileError, package_file, load_ros_parameters, resource, task_parameters


def prepare_configuration(profile, runtime_dir, scene_source, use_camera, demonstration_cameras=False):
    runtime = Path(runtime_dir)
    moveit = profile['moveit']
    file = lambda name: package_file(moveit['package'], moveit['files'][name], 'moveit.files.' + name)
    description = profile['robot_description']
    mappings = dict(description['mappings'])
    template = file('controllers_template').read_text()
    # This is controller YAML template rendering, never source-code rewriting.
    for key, value in mappings.items():
        template = template.replace('${' + key + '}', value)
    if '${' in template:
        raise ProfileError('moveit.files.controllers_template: unresolved template variables')
    controller_file = runtime / 'ros2_controllers.yaml'
    controller_file.write_text(template)
    mappings.update(use_gazebo='true', use_camera=str(use_camera).lower(), controllers_file=str(controller_file))
    if demonstration_cameras:
        if profile['robot_id'] != 'kinova_gen3_lite' or not use_camera:
            raise ProfileError('Three-view demonstration cameras require Kinova with use_camera=true')
        mappings['use_demonstration_cameras'] = 'true'
    model_file = package_file(description['package'], description['xacro_file'], 'robot_description.xacro_file')
    urdf = xacro.process_file(str(model_file), mappings=mappings).toxml()
    urdf_file = runtime / 'robot_description.urdf'
    urdf_file.write_text(urdf)
    model = ET.fromstring(urdf)
    links = {link.attrib['name'] for link in model.findall('link')}
    joints = {joint.attrib['name'] for joint in model.findall('joint')}
    planning = profile['planning']
    for field in ('ik_frame', 'base_frame'):
        if planning[field] not in links:
            raise ProfileError(f'planning.{field}={planning[field]!r}: not in generated robot model')
    srdf = ET.parse(file('semantic')).getroot()
    for joint in srdf.findall('.//joint'):
        if joint.attrib['name'] not in joints:
            raise ProfileError(f"moveit.files.semantic: unknown URDF joint {joint.attrib['name']!r}")
    controllers = yaml.safe_load(file('controllers').read_text())
    controller_map = controllers['moveit_simple_controller_manager']
    for name in controller_map['controller_names']:
        for joint in controller_map[name]['joints']:
            if joint not in joints:
                raise ProfileError(f'MoveIt controller {name}: unknown URDF joint {joint!r}')
    parameters = {
        'robot_description': urdf,
        'robot_description_semantic': file('semantic').read_text(),
        'robot_description_kinematics': yaml.safe_load(file('kinematics').read_text()),
        'robot_description_planning': {
            **yaml.safe_load(file('joint_limits').read_text()),
            **yaml.safe_load(file('cartesian_limits').read_text()),
        },
        'planning_pipelines': list(moveit['pipelines']),
        'default_planning_pipeline': planning['pipeline'],
        'moveit_manage_controllers': True,
        **controllers,
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'publish_robot_description': False,
        'publish_robot_description_semantic': True,
    }
    for name, relative in moveit['pipelines'].items():
        parameters[name] = yaml.safe_load(package_file(moveit['package'], relative, 'moveit.pipelines.' + name).read_text())
    scene = load_ros_parameters(resource(profile['scene'][scene_source], 'scene.' + scene_source))
    task = {**task_parameters(profile), **scene}
    perception = load_ros_parameters(resource(profile['perception']['params'], 'perception.params'))
    perception.update({key: value for key, value in profile['perception'].items() if key != 'params'})
    return parameters, task, perception, urdf_file, controller_file
