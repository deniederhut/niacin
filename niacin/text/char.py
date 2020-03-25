#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Character-based functions for enriching text data
"""

import collections
import json
from pkg_resources import resource_string
import typing

from scipy import random


LEETMAP = collections.OrderedDict(
    [
        ("anned", "&"),
        ("and", "&"),
        ("what", "wat"),
        ("are", "r"),
        ("ate", "8"),
        ("at", "@"),
        ("one", "1"),
        ("you", "u"),
        ("t", "7"),
        ("o", "0"),
        ("e", "3"),
        ("l", "1"),
    ]
)

CONTRACT = json.loads(
    resource_string("niacin", "data/contractions.json").decode("utf-8")
)
EXPAND = {v: k for k, v in CONTRACT.items()}
NEIGHBORS = json.loads(resource_string("niacin", "data/neighbors.json").decode("utf-8"))


def _sub_chars(string: str, probability: float, mapping: typing.Mapping) -> str:
    """Replace substrings with a given probability.

    Given a mapping, search string one by one for keys and replace with
    the appropriate value, with some probability. If your keys are not mutually
    exclusive (e.g. some part of them overlaps), the order in which they appear
    in the mapping becomes important.

    Args:
        string: text
        probability: probability of replacing a group of characters
        mapping: map of substring -> replacement

    Returns:
        enriched text
    """
    for pattern, sub in mapping.items():
        index = 0
        while 0 <= index < len(string):
            index = string.lower().find(pattern, index)
            if index < 0:
                break
            elif random.binomial(1, probability):
                string = string[:index] + sub + string[index + len(pattern) :]
                index += len(sub)
            else:
                index += len(pattern)
    return string


def add_fat_thumbs(string: str, p: float = 0.01) -> str:
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
            string = string[:index] + new_char + string[index + 1 :]
    return string


def add_contractions(string: str, p: float = 0.5) -> str:
    """Replace common word pairs with their contraction.

    This is done even when the contraction introduces ambiguity, as this is
    seen as preserving the semantics (arXiv:1812.04718_).

    Args:
        string: text
        p: probability of a word pair being replaced

    Returns:
        enriched text

    .. _arXiv:1812.04718 : https://arxiv.org/abs/1812.04718
    """
    return _sub_chars(string, probability=p, mapping=CONTRACT)


def add_expansions(string: str, p: float = 0.5) -> str:
    """Expand a contraction into individual tokens.

    See (arXiv:1812.04718_).

    Args:
        string: text
        p: probability of a word pair being replaced

    Returns:
        enriched text

    .. _arXiv:1812.04718 : https://arxiv.org/abs/1812.04718
    """
    return _sub_chars(string, probability=p, mapping=EXPAND)


def add_leet(string: str, p: float = 0.2) -> str:
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


def add_whitespace(string: str, p: float = 0.01) -> str:
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
    space = " "
    for index in range(len(string), -1, -1):
        if random.binomial(1, p):
            string = string[:index] + space + string[index:]
    return string


def remove_whitespace(string: str, p: float = 0.1) -> str:
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
    mapping = {" ": ""}
    return _sub_chars(string, probability=p, mapping=mapping)
