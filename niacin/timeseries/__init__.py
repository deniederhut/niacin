#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Functions for enriching timeseries data.

Includes transformations which operate in the time domain, and ones which
operate in the frequency domain. Organization of this module and basic set
of functionality are inspired by Wen et al.'s review, "Timeseries data
Augmentation for Deep Learning" (arXiv:2002.12478_).

Importable functions include:

Time domain
-----------

* add_slope_trend
* add_spike
* add_step_trend
* add_warp
* crop_and_stretch
* flip
* reverse


Frequency domain
----------------

* add_discrete_phase_shift
* add_high_frequency_noise
* add_random_frequency_noise
* remove_random_frequency

.. _arXiv:2002.12478 : https://arxiv.org/abs/2002.12478
"""

from .freq import (add_discrete_phase_shifts, add_high_frequency_noise, add_random_frequency_noise, remove_random_frequency)
from .time import (add_slope_trend, add_spike, add_step_trend, add_warp, crop_and_stretch, flip, reverse)