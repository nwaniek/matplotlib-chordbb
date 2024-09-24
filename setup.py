#!/usr/bin/env python

from setuptools import setup

setup(
    name='matplotlib-chordbb',
    version='0.1.0',
    description='Building blocks to make chord plots',
    author='Nicolai Waniek',
    author_email='n@rochus.net',
    py_modules=["chordbb"],
    install_requires=["numpy", "matplotlib"],
    license='MIT'
)
