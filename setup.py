#! /usr/bin/env python
from setuptools import setup

if __name__ == "__main__":
    setup(
        name='commtool',
        version='0.1.dev',
        description='Network Connectivity Toolbox in Python',
        long_description=open('README.md').read(),
        url='https://github.com/KordingLab/commtoolpy',
        author='Titipat Achakulvisut',
        author_email='my.titipat@gmail.com',
        license='(c) 2017 Titipat Achakulvisut',
        install_requires=['networkx', 'sklearn'],
        packages=['commtool']
    )
