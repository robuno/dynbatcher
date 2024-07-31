from setuptools import setup, find_packages

setup(
    name='dbstraining',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib'
    ],
    author='Unat Teksen',
    description='A package for dynamic batch size training in PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/robuno/dbs-training',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='MIT',  # Add this line to specify the license
)
