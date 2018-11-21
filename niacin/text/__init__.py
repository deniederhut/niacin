#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

"""
Functions for enriching text-like input data
"""

from .char import (
    add_fat_thumbs,
    add_leet,
    add_whitespace,
    remove_whitespace,
)
from .sentence import (
    add_applause,
    add_bytes,
    add_love,
)
from .word import (
    add_hypernyms,
    add_hyponyms,
    add_misspelling,
    add_parens,
    remove_articles,
)
