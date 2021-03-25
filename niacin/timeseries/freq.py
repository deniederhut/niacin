#!/usr/bin/env python
# -*- coding: utf-8

"""Frequency domain transforms
"""

import numpy as np
from numpy import random
from scipy.fft import irfft, rfft


def add_discrete_phase_shifts(x: np.ndarray, p: float=0.01, m: float=0.1):
    r"""Shift each frequency component with probability p by distance m*len(x).

    Compute real FFT on sequence. Then, for each frequency entry, with
    probability p, swap it with an entry that is ±m*len(x)//2 steps away.
    Swapped entries are tagged so that they are not moved twice. The 0th
    component is ignored.

    Args:
        x: sequence
        p: per-frequency probability of shifting
        m: magnitude of shift

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
        >>> ts.add_discrete_phase_shifts(x, 1.0, 0.1)
        |
        +1.03587-|      \     -|     \      \|     \      |
        |/|     | |    /|     ||     /|    / |    | |    /|
        |||     | |    ||     ||    | |    | |    | |    | |
        || |   |  |    | |   | |    |  |   | |    | |    | |
        |  |   |  |   |  |   |  |   |  |   | |    | |    | |
        |  |   |  |   |  |   |  |   |  |  |   |   | |   |  |
        +---|--|--|---|--|---|--|---|--|--|---|--|---|--|--|---+--
        +0  |  |   | |   |   |  |  |   |  |   |  |   |  |   +99
        |   | |    | |   |  |   |  |   |  |   |  |   |  |   |  *
        |   | |    | |    | |   |  |    | |   | |    |  |   | |
        |   | |    | |    | |    | |    | |    ||    |  |    ||
        |    \|    | |    \ |    | |    \|     ||     \|     |/
        -1.01053    /      /|     /      |     -|      |     -/
        |
    """
    fx = rfft(x)
    step = round(len(fx) * m)
    ps = random.binomial(1, p, len(fx))
    signs = random.binomial(1, 0.5, len(fx)) * 2 - 1
    # skip 0 Hz component
    for i in range(1, len(fx)):
        if ps[i] == 1:
            j = min(max(i + signs[i] * step, 0), len(fx)-1)
            if ps[j] == 1:
                fx[i], fx[j] = fx[j], fx[i]
                # don't shift the same component more than once
                ps[i], ps[j] = -1, -1
    return irfft(fx)


def add_random_frequency_noise(x: np.ndarray, p: float=0.01, m: float=0.1):
    r"""Add gaussian noise to each frequency with probability p and magnitude
    N(0,1) * m * max(fft(x))

    Args:
        x: sequence
        p: per-frequency probability of noise
        m: magnitude of noise

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
        >>> ts.add_random_frequency_noise(x, 0.5, 0.1)
        |
        +1.59613                                \|
        |  -\/\                 |               ||
        |\| / \              -|||             \/||\/|
        |/|    |             |/ |   \         |/  -/-\
        |      \         |   |  | ||\|        |      |        |
        +-------|-|-|--|-|-\\|--\|||-|-------|-------|-\------|+--
        +0      ||| |  | |/ /|   | | | |     |        /|    +99*
        |       ||| |  || /        | -/|     /         |      |
        |        |\ |\|                |\\-\|          | |   |
        |          / \|                //|| |           || \\|
        |             |                  ||             / || |
        |                                 |               |
        -1.90176                                          |
        |
    """
    fx = rfft(x)
    ps = random.binomial(1, p, len(fx))
    noise = random.randn(2*len(fx)).view(np.complex128) * m * fx.max()
    fx[1:] = fx[1:] + ps[1:] * noise[1:]
    return irfft(fx)


def add_high_frequency_noise(x: np.ndarray, p: float=0.01, m: float=0.1):
    r"""Add gaussian noise to single highest frequency component with
    probability p and magnitude N(0,1) * m * max(fft(x))

    Args:
        x: sequence
        p: per-sequence probability of noise
        m: magnitude of noise

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
        >>> ts.add_high_frequency_noise(x, 1.0, 1.0)
        |
        +2.11065              ||||             || ||
        | ||||||             ||||||           |||||||
        |||||||||           ||||||||         |||||||||
        ||||||||| |       ||||||||||        |||||||||||        *
        ||||||||||||     ||||||||||||      |||||||||||||      |
        |||||||||||||  ||||||||||||||| |  |||||||||||||||  | ||
        +||||||||||||||||||||||||||||||||||||||||||||||||||||||+--
        +0| |  |||||||||||||||  | |||||||||||||||  |||||||||+99
        ||      |||||||||||||      ||||||||||||     |||||||||||
        |        |||||||||||       ||||||||| |       ||||||||||
        |         |||||||||         ||||||||           ||||||||
        |          |||||||           ||||||             ||||||
        -2.11065    || ||             ||||               ||||
        |
    """
    fx = rfft(x)
    if random.binomial(1, p):
        noise = random.randn(2).view(np.complex128) * m * fx.max()
        fx[-1] = fx[-1] + noise
    return irfft(fx)


def remove_random_frequency(x: np.ndarray, p: float=0.01, m=None):
    r"""Remove each frequency component with probability p.

    Args:
        x: sequence
        p: per-frequency probability of removal
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
        +1
        |
        |
        |
        |
        |
        +------------------------------------------------------+--
        +0                                                  +99
        |
        |
        |
        |
        -1
    """
    fx = rfft(x)
    ps = random.binomial(1, 1-p, len(fx))
    fx = fx * ps
    return irfft(fx)