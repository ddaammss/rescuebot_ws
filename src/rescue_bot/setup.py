from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'rescue_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 런치 파일을 설치하기 위한 경로 설정
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chans',
    maintainer_email='ahwkt46@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'database = rescue_bot.database.srd_database_node:main',
            'srd_pose_emergency_node = rescue_bot.analyzer.srd_pose_emergency_node:main',
        ],
    },
)
