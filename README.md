# Network Community Toolbox in Python

Network Connectivity Toolbox in Python (still developing).
The original Matlab code snippets are available [here](http://commdetect.weebly.com/). The former Python version for Network Connectivity Toolbox is available [here](https://github.com/nangongwubu/Python-Version-for-Network-Community-Architecture-Toobox).


## Installation

Using `pip` to install directly from the repository

```bash
$ pip install git+git://github.com/KordingLab/nctpy.git
```

Or install by cloning from the repository

```bash
$ git clone https://github.com/KordingLab/nctpy
$ cd nctpy
$ python setup.py install
```


## Usage

Basically, we have functions equivalent to Network Community Toolbox.

```python
import nct
comm_ave_pairwise_spatial_dist = nct.comm_ave_pairwise_spatial_dist(partitions, locations)
```


## Requirements

- [numpy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [networkx](https://networkx.github.io/)


## Acknowledgement

This repository is developed during BE 566 (Network Neuroscience)
class at the University of Pennsylvania taught by [Prof. Danille Bassett](http://www.danisbassett.com/). The repository is developed
at [Konrad Kording lab](http://kordinglab.com/).
