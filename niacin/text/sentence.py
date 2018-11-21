#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Sentence-based functions for enriching text data
"""

import regex
from scipy import random


P_SPACE = regex.compile(r'(\s+)')


def add_applause(string, p=0.1):
    if random.binomial(1, p):
        string = P_SPACE.sub('\U0001f44f', string)
    return string

def add_bytes(string, p=0.1, length=100):
    if random.binomial(1, p):
        string = string + random.bytes(length).decode('utf-8', errors='replace')
    return string

def add_love(string, p=0.1):
    if random.binomial(1, p):
        string = string + ' love'
    return string
