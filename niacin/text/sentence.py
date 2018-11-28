#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Sentence-based functions for enriching text data
"""

import regex
from scipy import random


P_SPACE = regex.compile(r'(\s+)')


def add_applause(string: str, p: float=0.1) -> str:
    """Replace whitespace with clapping emojis.

    In online communities, replacing whitespace delimiters with the clapping
    emoji (\U0001f44f) is a way of indicating emphasis, possibly as a typographic
    replacement for the baton gesture. This has the unintended consequence
    of rendering word or token-based models ineffective.

    Args:
        string: text
        p: probability of replacing every whitespace character

    Returns:
        enriched text
    """
    if random.binomial(1, p):
        string = P_SPACE.sub('\U0001f44f', string)
    return string

def add_bytes(string: str, p: float=0.1, length: int=100) -> str:
    """Add random bytes to the end of a sentence.

    A common spam disguising technique includes appending random sequences of
    bytes to the end of text data. This can be effective against character
    based models, or loglinear models which include total length and character
    distribution as features. Random bytes are decoded as utf-8 with errors
    ignored, so the total number of characters will typically be smaller than
    the length input parameter.

    Args:
        string: text
        p: probability adding random bytes
        length: number of random bytes

    Returns:
        enriched text
    """
    if random.binomial(1, p):
        string = string + random.bytes(length).decode('utf-8', errors='replace')
    return string

def add_love(string: str, p: float=0.1) -> str:
    """Add love to the end of a sentence.

    Appends ``' love'`` to the end of a string. Including a word with large
    positive sentiment can be used to confuse sentiment-based filters for
    input data (citation needed).

    Args:
        string: text
        p: probability of adding ' love' to a sentence

    Returns:
        enriched text
    """
    if random.binomial(1, p):
        string = string + ' love'
    return string
