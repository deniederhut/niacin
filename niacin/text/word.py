#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Word-based functions for enriching text data
"""

import collections
import json
from pkg_resources import resource_string

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


def _sub_words(string, probability, mapping):
    words = string.split()
    for pattern, sub in mapping.items():
        for index, word in enumerate(words):
            if (word.lower() == pattern) and random.binomial(1, probability):
                words[index] = sub
    return ' '.join(word for word in words if word)

def add_hypernyms(string, p=0.01):
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in HYPERNYMS) and random.binomial(1, p):
            words[index] = random.choice(HYPERNYMS[word])
    return ' '.join(words)

def add_hyponyms(string, p=0.01):
    words = [WN.lemmatize(w) for w in string.split()]
    for index, word in enumerate(words):
        if (word in HYPONYMS) and random.binomial(1, p):
            words[index] = random.choice(HYPONYMS[word])
    return ' '.join(words)

def add_misspelling(string, p=0.1):
    words = string.split()
    for index, word in enumerate(words):
        if (word in MISSPELLINGS) and random.binomial(1, p):
            words[index] = random.choice(MISSPELLINGS[word])
    return ' '.join(words)

def add_parens(string, p=0.01):
    words = string.split()
    for index, word in enumerate(words):
        if random.binomial(1, p):
            words[index] = '(((' + word + ')))'
    return ' '.join(words)

def remove_articles(string, p=1.0):
    mapping = {article: '' for article in ARTICLES}
    return _sub_words(string, probability=p, mapping=mapping)
