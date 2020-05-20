#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import pytest

from niacin.text import sentence


@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("This is good.", 0.0, "This is good."),
        ("This is good.", 1.0, "This\U0001f44fis\U0001f44fgood."),
    ],
)
def test_add_applause(string, p, exp):
    res = sentence.add_applause(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,exp",
    [("", 0.0, ""), ("", 1.0, " love"), ("dog", 0.0, "dog"), ("dog", 1.0, "dog love")],
)
def test_add_love(string, p, exp):
    res = sentence.add_love(string, p)
    assert res == exp


@pytest.mark.parametrize(
    "string,p,length,exp",
    [
        ("", 0.0, 10, 0),
        ("", 1.0, 10, 10),
        ("dog", 0.0, 10, 3),
        ("dog", 1.0, 0, 3),
        ("dog", 1.0, 10, 13),
    ],
)
def test_add_bytes(string, p, length, exp):
    res = sentence.add_bytes(string, p, length)
    assert res[: len(string)] == string
    assert abs(len(res) - exp) < 5


@pytest.mark.slow
@pytest.mark.parametrize(
    "string,p,exp",
    [
        ("", 0.0, ""),
        ("", 1.0, ""),
        ("this is a test", 0.0, "this is a test"),
        ("this is a test", 1.0, "This is a test"),
        ("He asked if she said it", 0.0, "He asked if she said it"),
        ("He asked if she said it", 1.0, "He asked if she had said it"),
    ],
)
def test_backtranslation(string, p, exp):
    res = sentence.add_backtranslation(string, p)
    assert res == exp
