from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('VERSION') as f:
    version = f.read().strip()

setup(
    name='topic_inference',
    packages=find_packages(),
    version=version,
    install_requires=requirements
)