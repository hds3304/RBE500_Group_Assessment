import glob
import os
from setuptools import find_packages, setup

package_name = 'GA1'

py_files = glob.glob(os.path.join(package_name, "*.py"))
py_files = [os.path.basename(f)[:-3] for f in py_files if os.path.basename(f) != "__init__.py"]

entry_points = {f"{name} = {package_name}.{name}:main" for name in py_files}

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
    maintainer='harsh',
    maintainer_email='harsh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': list(entry_points),
    },
)
