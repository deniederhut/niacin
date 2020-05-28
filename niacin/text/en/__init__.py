#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Functions for enriching English language data.

Includes transformations which operate on characters, words, and whole
sentences. Importable functions include:

Character-based
---------------

* add_characters
* add_contractions
* add_fat_thumbs
* add_leet
* add_whitespace
* remove_characters
* remove_contractions
* remove_punctuation
* remove_whitespace
* swap_chars

Word-based
----------

* add_hypernyms
* add_hyponyms
* add_misspelling
* add_parens
* add_synonyms
* remove_articles
* swap_words

Sentence-based
--------------

* add_applause
* add_backtranslation
* add_bytes
* add_love
"""

from .char import (
    add_characters,
    add_contractions,
    remove_contractions,
    add_fat_thumbs,
    add_leet,
    add_whitespace,
    remove_characters,
    remove_punctuation,
    remove_whitespace,
    swap_chars,
)
from .sentence import add_applause, add_backtranslation, add_bytes, add_love
from .word import (
    add_hypernyms,
    add_hyponyms,
    add_misspelling,
    add_parens,
    add_synonyms,
    remove_articles,
    swap_words,
)
