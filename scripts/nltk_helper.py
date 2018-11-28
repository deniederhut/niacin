#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

from nltk import download


def main():
    for resource in ['wordnet']:
        download(resource)


if __name__ == '__main__':
    main()
