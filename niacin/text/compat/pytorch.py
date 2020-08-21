#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes for compatibility with PyTorch data loaders. Includes:

* MemoryTextDataset
* FileTextDataset
* DirectoryTextDataset
"""

import collections
import os
from pathlib import Path
import typing as t

from nltk import WordPunctTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab


class TextDatasetMixin:

    _data: t.Sequence
    _tokenizer: t.Callable
    _transforms: t.Iterable[t.Callable]
    _vocab: Vocab

    def _build_vocab(self) -> Vocab:
        counter: collections.Counter = collections.Counter()
        # older versions of torchtext require special symbols to be in the
        # vocabulary
        counter.update(["<unk>", "<pad>"])
        for doc in self._data:
            for token in self._tokenizer(doc):
                counter.update([token])
        return Vocab(counter)

    def _tokenize(self, text: str) -> t.List[str]:
        return self._tokenizer(text)

    def _transform(self, text: str) -> str:
        for transform in self._transforms:
            text = transform(text)
        return text

    def _vectorize(self, tokens: t.List[str]) -> torch.Tensor:
        return torch.tensor([self._vocab[token] for token in tokens])


class MemoryTextDataset(Dataset, TextDatasetMixin):
    """A text dataset for data that is already in memory. Accepts a list of
    text inputs and a list of labels.

    Args:
        data: list of text inputs
        labels: list of text labels
        transforms: either a list of transformation functions, or an Augmenter
            class
        tokenizer: a function that receives text and returns tokens
        vocab: a PyTorch Vocab object
    """

    def __init__(
        self,
        data: t.Sequence,
        labels: t.Sequence,
        transforms: t.Iterable[t.Callable] = None,
        tokenizer: t.Callable = None,
        vocab: Vocab = None,
    ):
        self._data = data
        self._labels = labels
        if transforms is None:
            self._transforms = []
        else:
            self._transforms = transforms
        if tokenizer is None:
            self._tokenizer = WordPunctTokenizer().tokenize  # type: ignore
        else:
            self._tokenizer = tokenizer  # type: ignore
        if vocab is None:
            self._vocab = self._build_vocab()
        else:
            self._vocab = vocab

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, t.Union[int, float]]:
        label = self._labels[index]
        data = self._data[index]
        data = self._transform(data)
        tokens = self._tokenize(data)
        vector = self._vectorize(tokens)
        return vector, label


class FileTextDataset(MemoryTextDataset):
    """A text dataset for data that is in a single file on disk. The reader
    assumes that the labels will be in the first column, and all text data
    will be in subsequent columns.

    Args:
        datafile: location of file on disk
        sep: delimiter between columns of data in file
        transforms: either a list of transformation functions, or an Augmenter
            class
        tokenizer: a function that receives text and returns tokens
        vocab: a PyTorch Vocab object
    """

    def __init__(
        self,
        datafile: str,
        sep: str = ",",
        transforms: t.Iterable[t.Callable] = None,
        tokenizer: t.Callable = None,
        vocab: Vocab = None,
    ):
        df = pd.read_table(datafile, sep=sep)
        self._data = df.iloc[:, 1:].apply(lambda s: s.str.cat(sep=" "), axis=1).tolist()
        self._labels = df.iloc[:, 0].tolist()
        if transforms is None:
            self._transforms = []
        else:
            self._transforms = transforms
        if tokenizer is None:
            self._tokenizer = WordPunctTokenizer().tokenize  # type: ignore
        else:
            self._tokenizer = tokenizer  # type: ignore
        if vocab is None:
            self._vocab = self._build_vocab()
        else:
            self._vocab = vocab


class DirectoryTextDataset(Dataset, TextDatasetMixin):
    """A text dataset for data that is spread across multiple files.

    Assumes two separate directories exist: one with text data, and one with
    labels, in separate files with the same names (but not necessarily the same
    extensions). E.g.

    * data/
        * review_one.txt
        * review_two.txt
        * ...
    * labels/
        * review_one.label
        * review_two.label
        * ...

    Args:
        data_dir: folder with data files
        labels_dir: folder with label files
        transforms: either a list of transformation functions, or an Augmenter
            class
        tokenizer: a function that receives text and returns tokens
        vocab: a PyTorch Vocab object
    """

    def __init__(
        self,
        data_dir: str,
        labels_dir: str,
        transforms: t.Iterable[t.Callable] = None,
        tokenizer: t.Callable = None,
        vocab: Vocab = None,
    ):
        self._data_dir = Path(data_dir)
        self._data = sorted(os.listdir(data_dir))
        self._labels_dir = Path(labels_dir)
        self._labels = sorted(os.listdir(labels_dir))
        if transforms is None:
            self._transforms = []
        else:
            self._transforms = transforms
        if tokenizer is None:
            self._tokenizer = WordPunctTokenizer().tokenize  # type: ignore
        else:
            self._tokenizer = tokenizer  # type: ignore
        if vocab is None:
            self._vocab = self._build_vocab()
        else:
            self._vocab = vocab

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self._labels[index]
        label_path = self._labels_dir / filename
        with open(label_path) as f:
            label = torch.tensor(float(f.read()))
        data_path = self._data_dir / filename
        with open(data_path) as f:
            data = f.read()
        data = self._transform(data)
        tokens = self._tokenize(data)
        vector = self._vectorize(tokens)
        return vector, label

    def _build_vocab(self) -> Vocab:
        counter: collections.Counter = collections.Counter()
        # older versions of torchtext require special symbols to be in the
        # vocabulary
        counter.update(["<unk>", "<pad>"])
        for filename in self._data:
            fp = self._data_dir / filename
            with open(fp) as f:
                doc = f.read()
            for token in self._tokenizer(doc):
                counter.update([token])
        return Vocab(counter)
