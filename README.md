# niacin

A Python library for replacing the missing variation in your text data.

[![PyPI version](https://badge.fury.io/py/niacin.svg)](https://badge.fury.io/py/niacin)
[![travis](https://travis-ci.org/deniederhut/niacin.svg?branch=master)](https://travis-ci.org/deniederhut/niacin/)
[![codecov](https://codecov.io/gh/deniederhut/niacin/branch/master/graph/badge.svg)](https://codecov.io/gh/deniederhut/niacin)
[![readthedocs](https://readthedocs.org/projects/niacin/badge/?version=latest)](https://niacin.readthedocs.io/en/latest/?badge=latest)


## Why should I use this?

Data collected for model training necessarily undersamples the likely
variance in the input space. This library is a collection of tools for
inserting typical kinds of perturbations to better approximate population
variance; and, for creating similar-but-incorrect examples to aid in
reducing the total size of the hypothesis space. These are commonly known as
<small>ENRICHMENT</small> and <small>NEGATIVE SAMPLING</small>, respectively.

## How do I use this?

Functions in niacin are separated into submodules for specific data types.
Functions expose a similar API, with two input arguments: the data to be
transformed, and the probability of applying a specific transformation.

enrichment:

```python
from niacin import text
data = "This is the song that never ends and it goes on and on my friends"
print(text.add_misspelling(data, p=1.0))
```

```output
This is teh song tath never ends adn it goes on anbd on my firends
```

negative sampling:

```python
from niacin import text
data = "This is the song that never ends and it goes on and on my friends"
print(text.add_hypernyms(data, p=1.0))
```

```output
This is the musical composition that never extremity and it exit on and on my person
```

## How do I install this?

with `pip`:

```sh
pip install niacin
```

from source:

```sh
git clone git@github.com:deniederhut/niacin.git && cd niacin && python setup.py install
```
