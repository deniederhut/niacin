#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    desc = f.read()

setup(
    name='niacin',
    version='0.1.1',
    packages=find_packages(),
    package_data={
        'niacin': ['data/*']
    },
    install_requires=[
        'nltk',
        'regex',
        'scipy',
    ],
    long_description=desc,
    long_description_content_type="text/markdown",
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
)
