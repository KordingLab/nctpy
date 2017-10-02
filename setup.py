#! /usr/bin/env python
from setuptools import setup

if __name__ == "__main__":
    setup(
        name='nctpy',
        version='0.1.dev',
        description='Network Connectivity Toolbox for Python',
        long_description=open('README.md').read(),
        url='https://github.com/KordingLab/commtoolpy',
        author='Titipat Achakulvisut',
        author_email='my.titipat@gmail.com',
        license='(c) 2017 Titipat Achakulvisut',
        install_requires=['numpy', 'networkx', 'sklearn'],
        packages=['nct']
    )
