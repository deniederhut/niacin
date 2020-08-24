#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    desc = f.read()

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

extras = {}
combined = set()
for extra in ('backtranslate', 'torch'):
    with open(extra + '-requirements.txt') as f:
        requirements = f.read().splitlines()
    extras[extra] = requirements
    combined.update(requirements)
extras['all'] = sorted(combined)


setup(
    name='niacin',
    version='0.4.0',
    packages=find_packages(),
    package_data={
        'niacin': ['data/*', 'py.typed']
    },
    python_requires=">=3.6",
    install_requires=install_requirements,
    extras_require=extras,
    long_description=desc,
    long_description_content_type="text/markdown",
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
)
