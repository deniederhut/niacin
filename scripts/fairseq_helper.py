#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

from niacin.text.en.sentence import _Translator


def main():
    """Download and cache the appropriate fairseq models"""
    _Translator.load_models()


if __name__ == '__main__':
    main()