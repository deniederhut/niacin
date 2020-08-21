#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


from functools import partial
import os
import pytest
import tempfile

from nltk import WordPunctTokenizer
import pandas as pd
import torch
from torchtext.vocab import Vocab

from niacin.text.compat.pytorch import (
    MemoryTextDataset,
    FileTextDataset,
    DirectoryTextDataset,
)
from niacin.text.en import char


def simple_file():
    data = pd.DataFrame({"labels": [0], "data": ["this is a test!"]})
    return data


@pytest.fixture
def temp_csv_file():
    data = simple_file()
    f = tempfile.NamedTemporaryFile(suffix=".csv", mode="w+")
    fp = f.name
    data.to_csv(fp, index=False)
    yield fp
    f.close()


@pytest.fixture
def temp_tsv_file():
    data = simple_file()
    f = tempfile.NamedTemporaryFile(suffix=".tsv", mode="w+")
    fp = f.name
    data.to_csv(fp, sep="\t", index=False)
    yield fp
    f.close()


@pytest.fixture
def temp_dir_with_files():
    data = simple_file()
    data_dir = tempfile.TemporaryDirectory()
    label_dir = tempfile.TemporaryDirectory()
    for index, row in data.iterrows():
        with open(os.path.join(data_dir.name, str(index) + ".txt"), "w") as f:
            f.write(row.iloc[1:].str.cat(sep=" "))
        with open(os.path.join(label_dir.name, str(index) + ".txt"), "w") as f:
            f.write(str(row.iloc[0]))
    yield data_dir.name, label_dir.name
    data_dir.cleanup()
    label_dir.cleanup()


class TestMemoryTextDataset:
    @pytest.mark.parametrize(
        "data,labels,l,i",
        [
            (["this is a test!"], [0], 1, 0),
            (["a bet", "is won", "by luck"], [1, 0, 1], 3, 2),
        ],
    )
    def test_indexing(self, data, labels, l, i):
        dataset = MemoryTextDataset(data, labels)
        assert len(dataset) == l
        assert dataset[i] is not None

    @pytest.mark.parametrize(
        "data,labels,transforms,expected",
        [
            (
                ["this is a test!"],
                [0],
                [partial(char.remove_whitespace, p=1.0)],
                "thisisatest!",
            ),
            (
                ["this is a test!"],
                [0],
                [
                    partial(char.remove_whitespace, p=1.0),
                    partial(char.remove_punctuation, p=1.0),
                ],
                "thisisatest",
            ),
        ],
    )
    def test_transforms(self, data, labels, transforms, expected):
        dataset = MemoryTextDataset(data, labels, transforms=transforms)
        result = dataset._transform(data[0])
        assert result == expected

    @pytest.mark.parametrize(
        "data,labels,tokenizer,expected",
        [
            (["this is a test!"], [0], None, ["this", "is", "a", "test", "!"]),
            (
                ["this is a test!"],
                [0],
                WordPunctTokenizer().tokenize,
                ["this", "is", "a", "test", "!"],
            ),
        ],
    )
    def test_tokenizer(self, data, labels, tokenizer, expected):
        dataset = MemoryTextDataset(data, labels, tokenizer=tokenizer)
        result = dataset._tokenize(data[0])
        assert result == expected

    @pytest.mark.parametrize(
        "data,labels,vocab,expected",
        [
            (["this is a test!"], [0], None, torch.tensor([6, 4, 3, 5, 2])),
            (
                ["this is a test!"],
                [0],
                Vocab({"this": 1, "<unk>": 1, "<pad>": 1}),
                torch.tensor([2, 0, 0, 0, 0]),
            ),
        ],
    )
    def test_getitem(self, data, labels, vocab, expected):
        dataset = MemoryTextDataset(data, labels, vocab=vocab)
        result, _ = dataset[0]
        assert torch.equal(result, expected)


class TestFileTextDataset:
    def test_indexing(self, temp_csv_file, temp_tsv_file):
        parameters = [(temp_csv_file, ",", 1, 0), (temp_tsv_file, "\t", 1, 0)]
        for fp, sep, l, i in parameters:
            dataset = FileTextDataset(fp, sep=sep)
            assert len(dataset) == l
            assert dataset[i] is not None

    def test_transforms(self, temp_csv_file, temp_tsv_file):
        parameters = [
            (
                temp_csv_file,
                ",",
                [partial(char.remove_whitespace, p=1.0)],
                "thisisatest!",
            ),
            (
                temp_tsv_file,
                "\t",
                [
                    partial(char.remove_whitespace, p=1.0),
                    partial(char.remove_punctuation, p=1.0),
                ],
                "thisisatest",
            ),
        ]
        for fp, sep, transforms, expected in parameters:
            dataset = FileTextDataset(fp, sep=sep, transforms=transforms)
            result = dataset._transform(dataset._data[0])
            assert result == expected

    def test_tokenizer(self, temp_csv_file, temp_tsv_file):
        parameters = [
            (temp_csv_file, ",", None, ["this", "is", "a", "test", "!"]),
            (
                temp_tsv_file,
                "\t",
                WordPunctTokenizer().tokenize,
                ["this", "is", "a", "test", "!"],
            ),
        ]
        for fp, sep, tokenizer, expected in parameters:
            dataset = FileTextDataset(fp, sep=sep, tokenizer=tokenizer)
            result = dataset._tokenize(dataset._data[0])
            assert result == expected

    def test_getitem(self, temp_csv_file, temp_tsv_file):
        parameters = [
            (temp_csv_file, ",", None, torch.tensor([6, 4, 3, 5, 2])),
            (
                temp_tsv_file,
                "\t",
                Vocab({"this": 1, "<unk>": 1, "<pad>": 1}),
                torch.tensor([2, 0, 0, 0, 0]),
            ),
        ]
        for fp, sep, vocab, expected in parameters:
            dataset = FileTextDataset(fp, sep=sep, vocab=vocab)
            result, _ = dataset[0]
            assert torch.equal(result, expected)


class TestDirectoryTextDataset:
    def test_indexing(self, temp_dir_with_files):
        data_dir, labels_dir = temp_dir_with_files
        parameters = [(data_dir, labels_dir, 1, 0)]
        for data, labels, l, i in parameters:
            dataset = DirectoryTextDataset(data, labels)
            assert len(dataset) == l
            assert dataset[i] is not None

    def test_transforms(self, temp_dir_with_files):
        data_dir, labels_dir = temp_dir_with_files
        parameters = [
            (
                data_dir,
                labels_dir,
                [partial(char.remove_whitespace, p=1.0)],
                "thisisatest!",
            ),
            (
                data_dir,
                labels_dir,
                [
                    partial(char.remove_whitespace, p=1.0),
                    partial(char.remove_punctuation, p=1.0),
                ],
                "thisisatest",
            ),
        ]
        for data, labels, transforms, expected in parameters:
            dataset = DirectoryTextDataset(data, labels, transforms=transforms)
            result = dataset._transform("this is a test!")
            assert result == expected

    def test_tokenizer(self, temp_dir_with_files):
        data_dir, labels_dir = temp_dir_with_files
        parameters = [
            (data_dir, labels_dir, None, ["this", "is", "a", "test", "!"]),
            (
                data_dir,
                labels_dir,
                WordPunctTokenizer().tokenize,
                ["this", "is", "a", "test", "!"],
            ),
        ]
        for data, labels, tokenizer, expected in parameters:
            dataset = DirectoryTextDataset(data, labels, tokenizer=tokenizer)
            result = dataset._tokenize("this is a test!")
            assert result == expected

    def test_getitem(self, temp_dir_with_files):
        data_dir, labels_dir = temp_dir_with_files
        parameters = [
            (data_dir, labels_dir, None, torch.tensor([6, 4, 3, 5, 2])),
            (
                data_dir,
                labels_dir,
                Vocab({"this": 1, "<unk>": 1, "<pad>": 1}),
                torch.tensor([2, 0, 0, 0, 0]),
            ),
        ]
        for data, labels, vocab, expected in parameters:
            dataset = DirectoryTextDataset(data, labels, vocab=vocab)
            result, _ = dataset[0]
            assert torch.equal(result, expected)
