#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Character-based functions for enriching text data
"""

import collections
import json
from pkg_resources import resource_string

from scipy import random


LEETMAP = collections.OrderedDict([
    ('anned', '&'),
    ('and', '&'),
    ('what', 'wat'),
    ('are', 'r'),
    ('ate', '8'),
    ('at', '@'),
    ('one', '1'),
    ('you', 'u'),
    ('t', '7'),
    ('o', '0'),
    ('e', '3'),
    ('l', '1'),
])

NEIGHBORS = json.loads(resource_string('niacin', 'data/neighbors.json'))


def _sub_chars(string, probability, mapping):
    for pattern, sub in mapping.items():
        index = 0
        while 0 <= index < len(string):
            index = string.lower().find(pattern, index)
            if index < 0:
                break
            elif random.binomial(1, probability):
                string = string[:index] + sub + string[index+len(pattern):]
                index += len(sub)
            else:
                index += len(pattern)
    return string

def add_fat_thumbs(string, p=0.01):
    for index, char in enumerate(string):
        if char in NEIGHBORS and random.binomial(1, p):
            new_char = random.choice(NEIGHBORS[char])
            string = string[:index] + new_char + string[index+1:]
    return string

def add_leet(string, p=0.2):
    return _sub_chars(string, probability=p, mapping=LEETMAP)

def add_whitespace(string, p=0.01):
    space = ' '
    for index in range(len(string), -1, -1):
        if random.binomial(1, p):
            string = string[:index] + space + string[index:]
    return string

def remove_whitespace(string, p=0.1):
    mapping = {' ': ''}
    return _sub_chars(string, probability=p, mapping=mapping)
