from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'force_estimation'

def package_files(directory):
    paths = []
    for path in glob(os.path.join(directory, '**'), recursive=True):
        if os.path.isfile(path):
            install_path = os.path.join('share', package_name, os.path.dirname(path))
            paths.append((install_path, [path]))
    return paths

data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/visualize_franka.launch.py']),
]

data_files += package_files('robots')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artuser',
    maintainer_email='ryo.hanai@aist.go.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo_force_estimation = force_estimation.demo_force_estimation:main',
        ],
    },
)
