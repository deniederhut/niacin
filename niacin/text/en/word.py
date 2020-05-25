#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Word-based functions for enriching English language data.

Importable functions include:

* add_hypernyms
* add_hyponyms
* add_misspelling
* add_parens
* add_synonyms
* remove_articles
* swap_words
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


def _get_wordnet():
    try:
        wn = wordnet.WordNetLemmatizer()
        wn.lemmatize("this is a test")
    except:
        print("Missing wordnet data -- attempting to download")
        import nltk

        nltk.download("wordnet")
        wn = wordnet.WordNetLemmatizer()
    return wn


def _sub_words(string: str, probability: float, mapping: typing.Mapping) -> str:
    """Replace words with a given probability.

    Split a string into words (the naïve way, on whitespace). Then, search
    word list for each key in the mapping, and replace it with its value with
    some probability. Then join them back together with a single whitespace.

    Args:
        string: text
        probability: probability of replacing a word
        mapping: map of substring -> replacement

    Returns:
        enriched text
    """
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
    wn = _get_wordnet()
    words = [wn.lemmatize(w) for w in string.split()]
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
    wn = _get_wordnet()
    words = [wn.lemmatize(w) for w in string.split()]
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
    wn = _get_wordnet()
    words = [wn.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in SYNONYMS) and random.binomial(1, p):
            words[index] = random.choice(SYNONYMS[word])
    return " ".join(words)


def remove_articles(string: str, p: float = 1.0) -> str:
    """Remove articles from text data.

    Matches and removes the following articles:

    * the
    * a
    * an
    * these
    * those
    * his
    * hers
    * their

    with probability p.

    Args:
        string: text
        p: probability of removing a given article

    Returns:
        enriched text
    """
    mapping = {article: "" for article in ARTICLES}
    return _sub_words(string, probability=p, mapping=mapping)


def swap_words(string: str, p: float = 0.01) -> str:
    """Swap adjacent words.

    With probability p, swap two adjacent words in a string. This preserves
    the vocabulary of input text while changing token order, and in
    theory should provide more of a challenge to recursive models than ones
    that rely on lexical distributions.

    .. note::
        to keep the interface consistent, niacin's implementation acts on
        a probability p, applied n-1 times, where n is the total number of words
        in the string. In the original paper (eda_), two words are chosen n times
        and swapped, where n is a count number given as a hyperparameter.

    Args:
        string: text
        p: probability of swapping two words

    Returns:
        enriched text

    .. _eda : https://arxiv.org/abs/1901.11196
    """
    words = string.split()
    index = 0
    while index < len(words) - 1:
        if random.binomial(1, p):
            words[index], words[index + 1] = words[index + 1], words[index]
            index += 2
        else:
            index += 1
    return " ".join(words)
