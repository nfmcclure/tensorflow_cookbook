#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('index.rst') as readme_file:
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
    name='TensorFlow Machine Learning',
    version='1.3.0',
    description="TensorFlow Machine Learning",
    author="Wei MEI",
    url='https://tensorflow-ml.org',
    
    license="MIT license",
    zip_safe=False,
    keywords='TensorFlow Machine Learning',
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
