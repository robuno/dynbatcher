from setuptools import setup, find_packages

setup(
    name='dbs-training',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.0.0',
        'torchvision>=0.3.0',
        'matplotlib>=3.0.0',
        'tqdm>=4.0.0'
    ],
    entry_points={
        'console_scripts': [
            'dbs-main=dbs_training.main:main'
        ]
    },
)
