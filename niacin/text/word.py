#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Word-based functions for enriching text data
"""

import collections
import json
from pkg_resources import resource_string
import typing

from scipy import random
from nltk import wordnet


HYPERNYMS = json.loads(resource_string("niacin", "data/hypernyms.json").decode("utf-8"))
HYPONYMS = json.loads(resource_string("niacin", "data/hyponyms.json").decode("utf-8"))
MISSPELLINGS = json.loads(
    resource_string("niacin", "data/misspellings.json").decode("utf-8")
)
SYNONYMS = json.loads(resource_string("niacin", "data/synonyms.json").decode("utf-8"))


ARTICLES = ("the", "a", "an", "these", "those", "his", "hers", "their")

WN = wordnet.WordNetLemmatizer()


def _sub_words(string: str, probability: float, mapping: typing.Mapping) -> str:
    words = string.split()
    for pattern, sub in mapping.items():
        for index, word in enumerate(words):
            if (word.lower() == pattern) and random.binomial(1, probability):
                words[index] = sub
    return " ".join(word for word in words if word)


def add_hypernyms(string: str, p: float = 0.01) -> str:
    """Replace word with a higher-level category.

    A common negative sampling technique involves replacing words
    in a sentence with a word that has the same general meaning, but
    is too general for the context, e.g.:

    "all dogs go to heaven" -> "all quadrupeds go to place"

    The replacement words are drawn from wordnet (wordnet_). For
    words with more than one possible replacement, one is selected using
    ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text

    .. _wordnet: https://wordnet.princeton.edu/
    """
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in HYPERNYMS) and random.binomial(1, p):
            words[index] = random.choice(HYPERNYMS[word])
    return " ".join(words)


def add_hyponyms(string: str, p: float = 0.01) -> str:
    """Replace word with a lower-level category.

    A common negative sampling technique involves replacing words
    in a sentence with a word that has the same general meaning, but
    is too specific for the context, e.g.:

    "all dogs go to heaven" -> "all Australian shepherds go to heaven"

    The replacement words are drawn from wordnet (wordnet_). For
    words with more than one possible replacement, one is selected using
    ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text

    .. _wordnet: https://wordnet.princeton.edu/
    """
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in HYPONYMS) and random.binomial(1, p):
            words[index] = random.choice(HYPONYMS[word])
    return " ".join(words)


def add_misspelling(string: str, p: float = 0.1) -> str:
    """Replace words with common misspellings.

    Replaces a word with a common way that word is mispelled, given one or
    more known, common misspellings taken from the Wikipedia spelling
    correction corpus (wikipedia_). For words with more than one common
    misspelling, one is chosen using ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text

    .. _wikipedia: https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings
    """
    words = string.split()
    for index, word in enumerate(words):
        if (word in MISSPELLINGS) and random.binomial(1, p):
            words[index] = random.choice(MISSPELLINGS[word])
    return " ".join(words)


def add_parens(string: str, p: float = 0.01) -> str:
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
            words[index] = "(((" + word + ")))"
    return " ".join(words)


def add_synonyms(string: str, p: float = 0.01) -> str:
    """Replace word with one that has a close meaning.

    A common data augmentation technique involves replacing words
    in a sentence with a word that has the same general meaning
    (arxiv:1509.01626_), e.g.:

    "all dogs go to heaven" -> "all domestic dog depart to heaven"

    The replacement words are drawn from wordnet (wordnet_). For
    words with more than one possible replacement, one is selected using
    ``random.choice``.

    Args:
        string: text
        p: conditional probability of replacing a word

    Returns:
        enriched text

    .. _arxiv:1509.01626 : https://arxiv.org/abs/1509.01626
    .. _wordnet: https://wordnet.princeton.edu/
    """
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in SYNONYMS) and random.binomial(1, p):
            words[index] = random.choice(SYNONYMS[word])
    return " ".join(words)


def remove_articles(string: str, p: float = 1.0) -> str:
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
    mapping = {article: "" for article in ARTICLES}
    return _sub_words(string, probability=p, mapping=mapping)
