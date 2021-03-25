#!/usr/bin/env python
# -*- coding: utf-8

"""Time domain transforms
"""


import numpy as np
from numpy import random
from scipy import interpolate



def add_slope_trend(x: np.ndarray, p: float=0.01, m: float=0.1) -> np.ndarray:
    r"""Add linear trend, with probability p, and magnitude m*std(x).

    The probability refers to the entire trend -- either it is added, or the
    original array is left alone. The direction of the trend is chosen randomly
    with probability=0.5.

    Args:
        x: sequence
        p: per-sequence probability of applying trend
        m: magnitude of trend

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.add_slope_trend(x, 1.0, 1.0)
        |
        +1.52476                                --\\
        |                     --\              /   \
        |   --\              // \\            //    \
        |  // \\             /    \           /      \
        | /    \            /     \          /       \         *
        | /     \           /      \         /        \       /
        |/      \          /       \\       /          \      /
        |/       \        /         \      /           \     /
        +---------\-------/----------\-----/------------\----/-+--
        +0        \      /           \    /             \\ /+99
        |          \    //            \\ //              --/
        |           \   /              --/
        -0.82       \--/
        |
    """
    if random.binomial(1, p):
        s = np.nanstd(x)
        sign = (random.binomial(1, 0.5, 1) * 2 - 1).item()
        x = x + np.linspace(0, sign*m*s, len(x))
    return x


def add_spike(x: np.ndarray, p: float=0.01, m: float=1.0) -> np.ndarray:
    r"""At each array entry, add a spike with probability p and magnitude
    m*std(x).

    The direction of the spike (up or down) is determined randomly with
    probability=0.5.

    Args:
        x: sequence
        p: per-entry probability of adding spike
        m: magnitude of spike

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.add_spike(x, 0.1, 1.0)
        |
        +1.61319              |
        | |                  ||
        ||| -|\              |--\               |-\
        ||// |\\             |  |            | /|  \
        ||   | \\           /   | \          |/ |   |
        ||   |  \           /   || \         |/ |   |\
        +--------\\-------||-----|-\\--------|------|\|--------+--
        +0        \      /||        \   |  //       | |     +99
        |          \     / |         | ||  /          |\\    /
        |           \  //            | ||//           ||\| ///
        |            --/             |||-/             | |-/
        |                            \||                 |
        -1.64856                      ||                 |
        |
    """
    s = np.nanstd(x)
    ps = random.binomial(1, p, len(x))
    signs = random.binomial(1, 0.5, len(x)) * 2 - 1
    x = x + ps * signs * (m * s)
    return x


def add_step_trend(x: np.ndarray, p: float=0.01, m: float=0.1) -> np.ndarray:
    r"""Add a stepwise trend, where each entry in the timeseries has p
    probability of a stepwise change of magnitude m*std(x).

    The direction of the stepwise trend is chosen with probability=0.5 for
    the entire trend (e.g. if the first step is upward, then every subsequent
    step is also upward).

    Args:
        x: sequence
        p: per-entry probability of step change
        m: magnitude of step change

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.add_step_trend(x, 0.05, 0.5)
        |
        +1 /--\              /--\
        | /    \\           //   |
        |/      \          //     \
        +--------\\-------/-------\\---------------------------+--
        +0        \\     /         \\         \ --\         +99
        |          \\   //          \\        //   \\
        |           \--/             |       /      \\
        |                             \   \//        \
        |                              --///          \\
        |                                              \\
        |                                               \\
        |                                                 -\ \/*
        -3.02032                                           ////
        |
    """
    s = np.nanstd(x)
    sign = (random.binomial(1, 0.5, 1) * 2 - 1).item()
    ps = random.binomial(1, p, len(x))
    x = x + np.cumsum(ps * (sign * m * s))
    return x


def add_warp(x: np.ndarray, p: float=0.01, m: float=0.1, interp_method: str = 'linear') -> np.ndarray:
    r"""Warp the distances between points in a timeseries.

    With probability p, upsample a timeseries by a scale of m*len(x). Then,
    randomly len(x) items from upsampled timeseries, where choice is uniform.

    Args:
        x: sequence
        p: per-sequence probability of warping
        m: magnitude of warp
        interp_method: scipy interp1d kind: can be ‘linear’, ‘nearest’,
            ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            or ‘next’. More details at scipy.interpolate.interp1d_.

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.add_warp(x, 1.0, 0.1)
        |
        +1 -\                ---\                 -\\
        | /  \|             |   --\              /  -\
        |-/   |             |     \             /     |
        ||    |            |      \\            |     |
        ||    -\          -|       |            |     |
        ||      |        /         |            |     |
        +-------|--------/---------|-----------/-------\------=+--
        +0      |       /           \         /        |    +99
        |        \     -/            \\       /        |     /
        |        \    |                \     /          |    /
        |         \   |                 \    /          |  //
        |          -  |                 \   |           |-/
        -1          -/                   -/-            --
        |

    .. _scipy.interpolate.interp1d: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    """
    old_size = len(x)
    step = round(old_size * m)
    if step < 2:
        # can't warp with no extra space
        return x
    if random.binomial(1, p):
        stretch_size = old_size * step
        old_t = np.linspace(0, 1, old_size)
        stretch_t = np.linspace(0, 1, stretch_size)
        select = np.sort(random.choice(stretch_t, size=old_size, replace=False))
        f = interpolate.interp1d(old_t, x, kind=interp_method)
        x = f(select)
    return x


def crop_and_stretch(x: np.ndarray,  p: float=0.01, m: float=0.1, interp_method: str = 'linear') -> np.ndarray:
    r"""Crop a sequence and stretch remaining entries to be the original size.

    With probability p, crop x such that it has m*len(x) entries remaining.
    Then, stretch it back to size len(x) using given interpolation method
    (default method is linear).

    Args:
        x: sequence
        p: per-sequence probability of cropping
        m: magnitude of crop (larger m is fewer entries from x)
        interp_method: scipy interp1d kind: can be ‘linear’, ‘nearest’,
            ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            or ‘next’. More details at scipy.interpolate.interp1d_.

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.crop_and_stretch(x, 1.0, 0.5)
        |
        +1---\                               -----\
        |     \\                           ///     \\
        |      \\                         //        \\
        |       \\                       /           \\
        |         \                     /              \
        |          \                   //               \
        +-----------\-----------------//-----------------\-----+--
        +0           \               /                   \\ +99
        |            \\             /                     \\
        |              \           //                      \\
        |               \         //                         \
        |                \\     ///                           \*
        -1                \----//
        |

    .. _scipy.interpolate.interp1d: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    """
    old_size = len(x)
    crop_size = round((1-m) * old_size)
    if (crop_size < 1) or (crop_size >= old_size):
        # if m << there might be no crop
        # if m >> the crop might be equal to length of sequence
        return x
    if random.binomial(1, p):
        start = random.randint(0, old_size-crop_size, 1).item()
        crop = x[start:start+crop_size]
        crop_t = np.linspace(0, 1, crop_size)
        new_t = np.linspace(0, 1, old_size)
        f = interpolate.interp1d(crop_t, crop, kind=interp_method)
        x = f(new_t)
    return x


def flip(x: np.ndarray, p: float=0.5, m=None) -> np.ndarray:
    r"""Flip sequence around origin with probability p.

    Args:
        x: sequence
        p: per-sequence probability of flipping
        m: ignored

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.flip(x, 1.0)
        |
        +1           --\               /-\               /-\
        |           /   \             //  \             //  \
        |          /    \            /    \             /    \
        |          /     \           /     \           /     \
        |         /      \          /      \           /      \
        |         /       \         /       \         /       \
        +--------/--------\--------/---------\-------/---------+--
        +0       /         \       /         \       /      +99
        |\      /           \      /          \      /
        | \     /           \     /           \     /
        | \    /             \    /            \    /
        |  \  //             \  //             \   /
        -1  -//               -//               --/
        |
    """
    if random.binomial(1, p):
        x = -x
    return x


def reverse(x: np.ndarray, p: float=0.5, m=None) -> np.ndarray:
    r"""Reverse order of sequence with probability p

    Args:
        x: sequence
        p: per-sequence probability of reversal
        m: ignored

    Returns:
        enriched sequence

    Examples:
        >>> x = np.sin(np.linspace(0, 6*np.pi, 100))
        |
        +1  -\\               -\\               --\
        |  /  \\             /  \\             /   \
        | /    \             /    \            /    \
        | /     \           /     \           /     \
        |/      \           /      \          /      \
        |/       \         /       \         /       \
        +--------\--------/--------\---------/-------\---------+--
        +0        \       /         \       /         \     +99
        |         \      /          \      /           \      /
        |          \     /           \     /           \     /
        |          \    /            \    /             \    /
        |           \   /             \\  /             \\  /
        -1           --/               \-/               \-/
        |
        >>> ts.reverse(x, 1.0)
        |
        +1           --\               /-\               /-\
        |           /   \             //  \             //  \
        |          /    \            /    \             /    \
        |          /     \           /     \           /     \
        |         /      \          /      \           /      \
        |         /       \         /       \         /       \
        +--------/--------\--------/---------\-------/---------+--
        +0       /         \       /         \       /      +99
        |\      /           \      /          \      /
        | \     /           \     /           \     /
        | \    /             \    /            \    /
        |  \  //             \  //             \   /
        -1  -//               -//               --/
    """
    if random.binomial(1, p):
        x = x[::-1]
    return x