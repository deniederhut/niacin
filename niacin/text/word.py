#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Word-based functions for enriching text data
"""

import collections
import json
from pkg_resources import resource_string
import typing

from scipy import random
from nltk import wordnet


HYPERNYMS = json.loads(resource_string('niacin', 'data/hypernyms.json'))
HYPONYMS = json.loads(resource_string('niacin', 'data/hyponyms.json'))
MISSPELLINGS = json.loads(resource_string('niacin', 'data/misspellings.json'))


ARTICLES = (
    'the',
    'a',
    'an',
    'these',
    'those',
    'his',
    'hers',
    'their',
)

WN = wordnet.WordNetLemmatizer()


def _sub_words(string: str, probability: float, mapping: typing.Mapping) -> str:
    words = string.split()
    for pattern, sub in mapping.items():
        for index, word in enumerate(words):
            if (word.lower() == pattern) and random.binomial(1, probability):
                words[index] = sub
    return ' '.join(word for word in words if word)

def add_hypernyms(string: str, p: float=0.01) -> str:
    """Replace word with a higher-level category.

    A common negative sampling technique involves replacing words
    in a sentence with a word that has the same general meaning, but
    is too general for the context, e.g.:

    "all dogs go to heaven" -> "all quadrupeds go to place"

    The replacement words are drawn from wordnet (citation needed). For
    words with more than one possible replacement, one is selected using
    ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text
    """
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in HYPERNYMS) and random.binomial(1, p):
            words[index] = random.choice(HYPERNYMS[word])
    return ' '.join(words)

def add_hyponyms(string: str, p: float=0.01) -> str:
    """Replace word with a lower-level category.

    A common negative sampling technique involves replacing words
    in a sentence with a word that has the same general meaning, but
    is too specific for the context, e.g.:

    "all dogs go to heaven" -> "all Australian shepherds go to heaven"

    The replacement words are drawn from wordnet (citation needed). For
    words with more than one possible replacement, one is selected using
    ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text
    """
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in HYPONYMS) and random.binomial(1, p):
            words[index] = random.choice(HYPONYMS[word])
    return ' '.join(words)

def add_misspelling(string: str, p: float=0.1) -> str:
    """Replace words with common misspellings.

    Replaces a word with a common way that word is mispelled, given one or
    more known, common misspellings taken from the Wikipedia spelling
    correction corpus (need citation). For words with more than one common
    misspelling, one is chosen using ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text
    """
    words = string.split()
    for index, word in enumerate(words):
        if (word in MISSPELLINGS) and random.binomial(1, p):
            words[index] = random.choice(MISSPELLINGS[word])
    return ' '.join(words)

def add_parens(string: str, p: float=0.01) -> str:
    """Wrap individual words in triple parentheses.

    Adds parentheses before and after a word, e.g. ``(((term)))``.
    This is a common tactic for disrupting tokenizers and other kinds
    of word based models.

    Args:
        string: text
        p: probability of wrapping a word

    Returns:
        enriched text
    """
    words = string.split()
    for index, word in enumerate(words):
        if random.binomial(1, p):
            words[index] = '(((' + word + ')))'
    return ' '.join(words)

def remove_articles(string: str, p: float=1.0) -> str:
    """Remove articles from text data.

    Matches and removes the following articles:
        'the',
        'a',
        'an',
        'these',
        'those',
        'his',
        'hers',
        'their',
    with probability p.

    Args:
        string: text
        p: probability of removing a given article

    Returns:
        enriched text
    """
    mapping = {article: '' for article in ARTICLES}
    return _sub_words(string, probability=p, mapping=mapping)
