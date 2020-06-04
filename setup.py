#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    desc = f.read()

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

setup(
    name='niacin',
    version='0.3.0',
    packages=find_packages(),
    package_data={
        'niacin': ['data/*', 'py.typed']
    },
    python_requires=">=3.6",
    install_requires=install_requirements,
    extras_require={
        'all': ['fairseq', 'fastbpe', 'sacremoses', 'torch'],
        'backtranslate': ['fairseq', 'fastbpe', 'sacremoses', 'torch']
    },
    long_description=desc,
    long_description_content_type="text/markdown",
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
)
