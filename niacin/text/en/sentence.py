#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Sentence-based functions for enriching English language data.

Importable functions include:

* add_applause
* add_backtranslation
* add_bytes
* add_love
"""

import regex
from scipy import random
import warnings


P_SPACE = regex.compile(r"(\s+)")


class _Translator:
    """Wrapper around fairseq language models (arXiv:1904.01038_).

    On first initialization, the instance loads language models and stores
    them as attributes on the class. New instances after this do not reload
    them. Currently implements translation from English to German, and the
    reverse.

    Attributes
    ----------
    en2de: callable
        translate from English to German
    de2en: callable
        translate from German to English
    translators: dict
        mapping of model names to model objects

    .. _arXiv:1904.01038 : https://arxiv.org/abs/1904.01038
    """

    translators: dict = {}

    def __init__(self):
        self.load_models()
        self.en2de = self.translators["en2de"].translate
        self.de2en = self.translators["de2en"].translate

    @classmethod
    def load_models(cls, force: bool = False):
        warnings.warn(
            "Backtranslation uses large translation models (~6GB) and can "
            "hours to download on the first use."
        )
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch not found - you may need to install extras with"
                "'pip install niacin[all]'"
            )
        if force or "en2de" not in cls.translators:
            cls.translators["en2de"] = torch.hub.load(
                "pytorch/fairseq",
                "transformer.wmt19.en-de.single_model",
                tokenizer="moses",
                bpe="fastbpe",
            )
        if force or "de2en" not in cls.translators:
            cls.translators["de2en"] = torch.hub.load(
                "pytorch/fairseq",
                "transformer.wmt19.de-en.single_model",
                tokenizer="moses",
                bpe="fastbpe",
            )
        # turn off dropout
        for name, model in cls.translators.items():
            model.eval()

    def backtranslate(self, string: str) -> str:
        return self.de2en(self.en2de(string))


def add_applause(string: str, p: float = 0.1) -> str:
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
        string = P_SPACE.sub("\U0001f44f", string)
    return string


def add_bytes(string: str, p: float = 0.1, length: int = 100) -> str:
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
        string = string + random.bytes(length).decode("utf-8", errors="replace")
    return string


def add_love(string: str, p: float = 0.1) -> str:
    """Add love to the end of a sentence.

    Appends ``' love'`` to the end of a string. Including a word with large
    positive sentiment can be used to confuse sentiment-based filters for
    input data (arXiv:1808.0911_).

    Args:
        string: text
        p: probability of adding ' love' to a sentence

    Returns:
        enriched text

    .. _arXiv:1808.0911 : https://arxiv.org/abs/1808.09115
    """
    if random.binomial(1, p):
        string = string + " love"
    return string


def add_backtranslation(string: str, p: float = 0.5) -> str:
    """Translate a sentence into another language and back.

    Use a fairseq model to translate a sentence from Enligh into German,
    then translate the German back into English with another fairseq model
    (arXiv:1904.01038_). Anecdotally, this generates sequences with similar
    semantic content, but different word choices, and is a popular way to
    augment small datasets in high resource languages (arXiv:1904.12848_).

    .. warning::
        Backtranslation uses large neural machine translation (NMT)
        models. The first time you call this function, it will download
        and cache up to 6GB of data, which can take hours depending on your
        connection speed. The slowness only happens once, but the model size
        will impact memory usage every time you use this function.

    Args:
        string: text
        p: probability of backtranslating a sentence

    Returns:
        enriched text

    .. _arXiv:1904.01038 : https://arxiv.org/abs/1904.01038
    .. _arXiv:1904.12848 : https://arxiv.org/abs/1904.12848

    """
    # the fairseq models do weird stuff with empty strings
    if not string:
        return string
    if random.binomial(1, p):
        t = _Translator()
        string = t.backtranslate(string)
    return string
