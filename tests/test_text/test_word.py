#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import pytest

from niacin.text import word


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("The man has a brown dog", 0.0, "The man has a brown dog"),
        ("The man has a brown dog", 1.0, "man has brown dog"),
    ],
)
def test_remove_articles(string, p, exp):
    res = word.remove_articles(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        (
            "politician persuades dramatic rhythms",
            0.0,
            "politician persuades dramatic rhythms",
        ),
        (
            "politician persuades dramatic rhythms",
            1.0,
            "politican pursuades dramtic rythyms",
        ),
    ],
)
def test_add_misspellings(string, p, exp):
    res = word.add_misspelling(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("It has a feud", 0.0, "It ha a feud"),
        ("It has a feud", 1.0, "It ha a vendetta"),
    ],
)
def test_add_hyponyms(string, p, exp):
    res = word.add_hyponyms(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("It has a sore", 0.0, "It ha a sore"),
        ("It has a sore", 1.0, "It ha a infection"),
    ],
)
def test_add_hypernyms(string, p, exp):
    res = word.add_hypernyms(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("It is computable", 0.0, "It is computable"),
        ("It is computable", 1.0, "It is estimable"),
    ],
)
def test_add_synonyms(string, p, exp):
    res = word.add_synonyms(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [("", 0.0, ""), ("", 1.0, ""), ("dog", 0.0, "dog"), ("dog", 1.0, "(((dog)))")],
)
def test_add_parens(string, p, exp):
    res = word.add_parens(string, p)
    assert res == exp

@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("The man has a brown dog", 0.0, "The man has a brown dog"),
        ("The man has a brown dog", 1.0, "man The a has dog brown"),
    ],
)
def test_swap_words(string, p, exp):
    res = word.swap_words(string, p)
    assert res == exp