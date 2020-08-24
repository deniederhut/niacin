#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import pytest

from niacin.augment import RandAugment
from niacin.text.en import add_whitespace, remove_whitespace


class TestRandAugment:
    @pytest.mark.parametrize('n', [0, 1, 2])
    def test_returns_n_transforms(self, n):
        funs_in = [lambda x, p: x] * 10
        rand = RandAugment(funs_in, n=n)
        funs_out = list(rand)
        assert len(funs_out) == n

    def test_raises_on_large_n(self):
        with pytest.raises(ValueError):
            rand = RandAugment([])

    @pytest.mark.parametrize('inp,exp', [
        (-1, 0),
        (0, 0),
        (10, 10),
        (100, 100),
        (101, 100)
    ])
    def test_bounds_on_m(self, inp, exp):
        funs = [lambda x, p: x] * 10
        rand = RandAugment(funs, m=inp)
        assert rand.m == exp

    @pytest.mark.parametrize('m, inp,exp', [
        (0, "this is a test", "this is a test"),
        (100, "this is a test", "thisisatest"),
    ])
    def test_transforms(self, m, inp, exp):
        funs = [add_whitespace, remove_whitespace]
        rand = RandAugment(funs, n=2, m=m, shuffle=False)
        for transform in rand:
            inp = transform(inp)
        assert inp == exp