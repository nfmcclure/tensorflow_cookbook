#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

#requirements = [
#    'MLMoleSci>=0.3.0'
#]

#test_requirements = [
#    'tox',
#    'flake8'
#]

setup(
    name='Machine Learning in Molecular Sciences',
    version='1.3.0',
    description="Machine Learning in Molecular Sciences",
    author="Wei MEI",
    url='https://Machine-Learning-in-the-Molecular-Sciences.ml',
    
    license="MIT license",
    zip_safe=False,
    keywords='Machine-Learning-in-Molecular-Sciences',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite='tests'
)
