from setuptools import setup
from setuptools import find_packages

## README
with open("README.md", "r") as fh:
    long_description = fh.read()

## Requirements
with open("requirements.txt", "r") as r:
    requirements = [i.strip() for i in r.readlines()]

## Run Setup
setup(
    name="stigma",
    version="0.0.1",
    author="Multiple",
    description="Stigmatizing Language in Medical Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=requirements,
)