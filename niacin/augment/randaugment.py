#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Implementation of RandAugment algorithm
"""

from functools import partial
import typing as t

from numpy.random import default_rng


class RandAugment:
    """Implements RandAugment algorithm (randaugment_) as an iterator.
    RandAugment selects ``n`` functions at random from a sequence, and
    initializes them with magnitude ``m``.

    To use it, initialize this class with a list of transformation functions.
    Every time it is iterated over (e.g. in a for loop) it yields a random
    subset of those transformation functions.

    The original paper claims that m is on a scale from 0-10, but its listed
    experiments regularly use magnitudes in the 20s and 30s. We take this to
    mean that "10" was a typo, and the scale was meant to extend to 100.

    The defaults for m and n have been set to 1 and 10, respectively, for
    safety. Depending on your model size and task, you may achieve more
    accurate results with n ∈ {2, 3} and an m in the range [10, 20). By
    default, the transforms in a single sample will be returned in random
    order. If your transforms must occur in a logical sequence (e.g. swapping
    synonyms before removing random characters), set shuffle to False.

    Args:
        transforms: sequence of transformation functions
        m: magnitude of transformation, on a scale of 0-100
        n: number of transforms to select
        shuffle: return transforms in random order
        seed: seed to use for the random number generator

    .. _randaugment: https://arxiv.org/abs/1909.13719
    """

    # niacin functions use floats for magnitude
    _p: float = 0.0

    def __init__(
        self,
        transforms: t.List[t.Callable],
        m: int = 10,
        n: int = 1,
        shuffle: bool = True,
        seed: int = None
    ):
        self._transforms = list(transforms)
        self.n = n
        self.m = m
        self._shuffle = shuffle
        self._rng = default_rng(seed=seed)

    def __len__(self):
        return len(self._transforms)

    def __iter__(self):
        choices = [
            partial(fun, p=self._p) for fun in self._rng.choice(
                self._transforms, size=self._n, replace=False, shuffle=self._shuffle
            )
        ]
        return iter(choices)

    @property
    def n(self) -> int:
        """Size of sample - should be less than total available transforms
        """
        return self._n

    @n.setter
    def n(self, value: int):
        n_funs = len(self._transforms)
        if value > n_funs:
            msg = f"Sample size n={value} must be <= number of transforms={n_funs}"
            raise ValueError(msg)
        self._n = value

    @property
    def m(self) -> int:
        """Magnitude of transformation - should be a number between 0 and 100
        """
        return int(self._p * 100)

    @m.setter
    def m(self, value: int):
        self._p = min(max(value / 100, 0.0), 1.0)