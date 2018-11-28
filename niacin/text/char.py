#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Character-based functions for enriching text data
"""

import collections
import json
from pkg_resources import resource_string
import typing

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


def _sub_chars(string: str, probability: float, mapping: typing.Mapping) -> str:
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

def add_fat_thumbs(string: str, p: float=0.01) -> str:
    """Replace characters with QWERTY neighbors.

    One source of typographic mistakes comes from pressing a nearby key
    on a keyboard (or on a touchscreen). With probability p, replace each
    character is a string with one from a set of its neighbors. The
    replacement is chosen using ``random.choice``.

    Args:
        string: text
        p: probability of replacing a character

    Returns:
        enriched text
    """
    for index, char in enumerate(string):
        if char in NEIGHBORS and random.binomial(1, p):
            new_char = random.choice(NEIGHBORS[char])
            string = string[:index] + new_char + string[index+1:]
    return string

def add_leet(string: str, p: float=0.2) -> str:
    """Replace character groups with visually or aurally similar ones.

    Character groups given in ``LEETMAP.keys()`` are searched for in
    priority (roughly from largest to smallest), and are replaced with
    some associated value with probability p. E.g.:

    | "Hello, you are banned"
    | "Hello, you are b&"
    | "Hello, you r b&"
    | "Hello, u r b&"
    | "H3110, u r b&"

    Args:
        string: text
        p: condtional probability of replacing a character group

    Returns:
        enriched text
    """
    return _sub_chars(string, probability=p, mapping=LEETMAP)

def add_whitespace(string: str, p: float=0.01) -> str:
    """Remove a spacebar characters with probability p.

    Selective removal of whitespace can be reduce the effectiveness of word-
    based models, or those which depend on word tokenizers as part of the
    data pipeline.

    Args:
        string: text
        p: probability of removing a space character

    Returns:
        enriched text
    """
    space = ' '
    for index in range(len(string), -1, -1):
        if random.binomial(1, p):
            string = string[:index] + space + string[index:]
    return string

def remove_whitespace(string: str, p: float=0.1) -> str:
    """Remove a spacebar characters with probability p.

    Selective removal of whitespace can be reduce the effectiveness of word-
    based models, or those which depend on word tokenizers as part of the
    data pipeline.

    Args:
        string: text
        p: probability of removing a space character

    Returns:
        enriched text
    """
    mapping = {' ': ''}
    return _sub_chars(string, probability=p, mapping=mapping)
