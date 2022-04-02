#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Character-based functions for enriching English language data.

Importable functions include:

* add_characters
* add_contractions
* add_fat_thumbs
* add_leet
* add_macbook_keyboard
* add_whitespace
* remove_characters
* remove_contractions
* remove_punctuation
* remove_whitespace
* swap_chars
"""

import collections
import json
from pkg_resources import resource_string
from string import ascii_letters, punctuation
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


def add_characters(string: str, p: float = 0.01) -> str:
    """Insert individual characters with probability p.

    These are chosen randomly from the ascii alphabet (including
    both upper and lower cases).

    Args:
        string: text
        p: probability of removing a character

    Returns:
        enriched text
    """
    for index in reversed(range(len(string))):
        if random.binomial(1, p):
            new_char = random.choice(list(ascii_letters))
            string = string[:index] + new_char + string[index:]
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


def remove_contractions(string: str, p: float = 0.5) -> str:
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


def add_macbook_keyboard(string: str, p: float = 0.1) -> str:
    """Repeats or removes each character with probability p.

    Bad keyboards can be a common source of typographical errors by
    repeating characters or by omitting them, e.g. because the individual
    keys get stuck. With probability p, we modify a character, with a
    50/50 chance of either removing it, or repeating it twice.

    Args:
        string: text
        p: probability of changing letter count

    Returns:
        enriched text
    """
    for index in reversed(range(len(string))):
        if random.binomial(1, p):
            count = random.choice([0, 2])
            string = string[:index] + string[index]*count + string[index+1:]
    return string


def add_whitespace(string: str, p: float = 0.01) -> str:
    """Add a spacebar character with probability p.

    Extraneous whitespace, especially when it occurs in the middle of an
    important word, can be reduce the effectiveness of models which depend
    on word tokenizers as part of the data pipeline.

    Args:
        string: text
        p: probability of adding a space character

    Returns:
        enriched text
    """
    space = " "
    for index in range(len(string), -1, -1):
        if random.binomial(1, p):
            string = string[:index] + space + string[index:]
    return string


def remove_characters(string: str, p: float = 0.01) -> str:
    """Remove individual characters with probability p.

    Args:
        string: text
        p: probability of removing a character

    Returns:
        enriched text
    """
    for index in reversed(range(len(string))):
        if random.binomial(1, p):
            string = string[:index] + string[index + 1 :]
    return string


def remove_punctuation(string: str, p: float = 0.25) -> str:
    """Remove punctuation with probability p.

    The removal of punctuation is a common data cleaning step for fast but
    high bias models and data processing algorithms. When that punctuation
    occurs in the middle of the word (e.g. indicating possessiveness), its
    removal may change the semantics of the string.

    Args:
        string: text
        p: probability of removing punctuation

    Returns:
        enriched text
    """
    mapping = {k: "" for k in punctuation}
    return _sub_chars(string, probability=p, mapping=mapping)


def remove_whitespace(string: str, p: float = 0.1) -> str:
    """Remove a spacebar character with probability p.

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


def swap_chars(string: str, p: float = 0.05) -> str:
    """Swap adjacent characters.

    With probability p, swap two adjacent characters in a string. No
    character gets swapped more than once, so cannot end up in any locations
    that are not adjacent to its starting position.

    .. note::
        to keep the interface consistent, niacin's implementation acts on
        a probability p, applied n-1 times, where n is the total number of
        characters in the string. The implementation in noisemix_ (called
        ``flip_chars``) chooses two letters at random and exchanges their
        positions, exactly once per string.

    Args:
        string: text
        p: probability of swapping two characters

    Returns:
        enriched text

    .. _noisemix : https://github.com/noisemix/noisemix
    """
    chars = list(string)
    index = 0
    while index < len(chars) - 1:
        if random.binomial(1, p):
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            index += 2
        else:
            index += 1
    return "".join(chars)
