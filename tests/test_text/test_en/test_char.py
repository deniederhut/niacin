#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import pytest
from unittest.mock import patch

from niacin.text.en import char


@pytest.mark.parametrize(
    "string,p,l", [("", 0.0, 0), ("", 1.0, 0), ("bob", 0.0, 3), ("bob", 1.0, 6)]
)
def test_add_characters(string, p, l):
    res = char.add_characters(string, p)
    assert len(res) == l


@pytest.mark.parametrize("string", [(""), ("qwerty")])
def test_add_fat_thumbs(string):
    res = char.add_fat_thumbs(string, 0.0)
    assert res == string
    res = char.add_fat_thumbs(string, 1.0)
    for left, right in zip(res, string):
        assert left != right


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("you what mate?", 0.0, "you what mate?"),
        ("you what mate?", 1.0, "u w@ m8?"),
        ("shadow banned", 1.0, "shad0w b&"),
    ],
)
def test_add_leet(string, p, exp):
    res = char.add_leet(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("alice is not dead", 0.0, "alice is not dead"),
        ("alice is not dead", 1.0, "alice isn't dead"),
    ],
)
def test_add_contractions(string, p, exp):
    res = char.add_contractions(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("alice isn't dead", 0.0, "alice isn't dead"),
        ("alice isn't dead", 1.0, "alice is not dead"),
    ],
)
def test_add_expansionss(string, p, exp):
    res = char.remove_contractions(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,l", [("", 0.0, 0), ("", 1.0, 0), ("bob", 0.0, 3), ("bob", 1.0, 0)]
)
def test_remove_characters(string, p, l):
    res = char.remove_characters(string, p)
    assert len(res) == l


@pytest.mark.parametrize(
    "string,p,l",
    [
        ("", 0.0, 0),
        ("", 1.0, 0),
        (r"""~`!'";:,.<>[]\_-""", 0.0, 16),
        (r"""~`!'";:,.<>[]\#$""", 1.0, 0),
        (r"""bob~`!'";:,.<>[]\@&""", 0.0, 19),
        (r"""bob~`!'";:,.<>[]\{}""", 1.0, 3),
    ],
)
def test_remove_punctuation(string, p, l):
    res = char.remove_punctuation(string, p)
    assert len(res) == l


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("The man has a brown dog", 0.0, "The man has a brown dog"),
        ("The man has a brown dog", 1.0, "Themanhasabrowndog"),
    ],
)
def test_remove_whitespace(string, p, exp):
    res = char.remove_whitespace(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [("", 0.0, ""), ("", 1.0, " "), ("dog", 0.0, "dog"), ("dog", 1.0, " d o g ")],
)
def test_add_whitespace(string, p, exp):
    res = char.add_whitespace(string, p)
    assert res == exp

@pytest.mark.parametrize(
    "string,p,choice,exp",
    [
        ("", 0.0, 0, ""),
        ("", 1.0, 2, ""),
        ("haircut", 0.0, 0, "haircut"),
        ("haircut", 1.0, 0, ""),
        ("haircut", 0.0, 2, "haircut"),
        ("haircut", 1.0, 2, "hhaaiirrccuutt"),
    ],
)
def test_add_macbook_keyboard(string, p, choice, exp):
    with patch('niacin.text.en.char.random.choice', return_value=choice) as mock:
        res = char.add_macbook_keyboard(string, p)
        assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("The man", 0.0, "The man"),
        ("The man", 1.0, "hT eamn"),
    ],
)
def test_swap_chars(string, p, exp):
    res = char.swap_chars(string, p)
    assert res == exp