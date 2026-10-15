from setuptools import find_packages, setup

package_name = 'bme_ros2_tutorials_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skylerkuo',
    maintainer_email='skylerkuo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'py_hello_world = bme_ros2_tutorials_py.hello_world:main',
            'py_publisher = bme_ros2_tutorials_py.publisher:main',
            'py_publisher_oop = bme_ros2_tutorials_py.publisher_oop:main',
            'py_subscriber = bme_ros2_tutorials_py.subscriber:main',
            'py_publisher_param = bme_ros2_tutorials_py.publisher_parameter:main',
            'py_service_server = bme_ros2_tutorials_py.service_server:main',
            'py_service_client = bme_ros2_tutorials_py.service_client:main',
            'py_multithread_demo = bme_ros2_tutorials_py.multithread_node:main',
            'py_action_server = bme_ros2_tutorials_py.action_server:main',
            'py_action_client = bme_ros2_tutorials_py.action_client:main',
        ],
    },
)
