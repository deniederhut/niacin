#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import pytest

import numpy as np
from scipy.fft import rfft

from niacin.timeseries import time


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_add_slope_trend(x, p, m):
    res = time.add_slope_trend(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res[1:], x[1:]).all()
        diff = res - x
        assert diff[0] == 0
        assert diff.sum() != 0
        assert np.abs(diff)[-1] == np.nanstd(x)


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_add_spike(x, p, m):
    res = time.add_spike(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).all()
        diff = res - x
        assert diff.sum() != 0
        assert (np.isclose(np.abs(diff), np.nanstd(x))).all()


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_add_step_trend(x, p, m):
    res = time.add_step_trend(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).all()
        diff = res - x
        assert diff.sum() != 0
        assert np.isclose(np.abs(diff)[0], np.nanstd(x))
        assert np.isclose(np.abs(diff)[1], 2*np.nanstd(x))
        assert np.isclose(np.abs(diff)[2], 3*np.nanstd(x))


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.5),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_add_warp(x, p, m):
    res = time.add_warp(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).any()
        assert all((x.min() < res) & (res < x.max()))


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.5),
    ]
)
def test_crop_and_stretch(x, p, m):
    res = time.crop_and_stretch(x, p, m)
    assert len(res) == len(x)
    if (p == 0.0) or (m == 0.0):
        np.testing.assert_almost_equal(res, x)
    elif (p == 1.0) and (m == 1.0):
        np.testing.assert_almost_equal(res, x)
    else:
        assert np.not_equal(res, x).any()


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_flip(x, p, m):
    res = time.flip(x, p, m)
    assert len(res) == len(x)
    if p == 0.0:
        np.testing.assert_almost_equal(res, x)
    else:
        np.testing.assert_almost_equal(-1*res, x)


@pytest.mark.parametrize(
    "x,p,m", [
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 0.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 0.0, 1.0),
        (np.sin(np.linspace(0, 6*np.pi, 100)), 1.0, 1.0),
    ]
)
def test_reverse(x, p, m):
    res = time.reverse(x, p, m)
    assert len(res) == len(x)
    if p == 0.0:
        np.testing.assert_almost_equal(res, x)
    else:
        np.testing.assert_almost_equal(res[::-1], x)