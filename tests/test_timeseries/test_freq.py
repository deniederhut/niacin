#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import pytest

import numpy as np
from scipy.fft import rfft

from niacin.timeseries import freq


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.02),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.01),
    ]
)
def test_add_discrete_phase_shifts(x, p, m):
    res = freq.add_discrete_phase_shifts(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).all()
        f_res = rfft(res)
        assert f_res.argmax() in (2, 4)


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.5),
    ]
)
def test_add_high_frequency_noise(x, p, m):
    res = freq.add_high_frequency_noise(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).all()
        f_x = rfft(x)
        f_res = rfft(res)
        np.testing.assert_almost_equal(f_res[:-1], f_x[:-1])
        assert f_res[-1] != f_x[-1]


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_add_random_frequency_noise(x, p, m):
    res = freq.add_random_frequency_noise(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).all()
        f_x = rfft(x)
        f_res = rfft(res)
        assert np.not_equal(res, x).all()


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_remove_random_frequency(x, p, m):
    res = freq.remove_random_frequency(x, p, m)
    assert len(res) == len(x)
    if p == 0.0:
        np.testing.assert_almost_equal(res, x)
    else:
        assert res.sum() == 0
