.. _with-pytorch:

using niacin with pytorch loaders
=================================

In PyTorch, a common practice is to interact with datasets via a high level ``Dataset`` class. This makes it easy to, for example, apply augmentations to image data on the fly, by passing a list of transformations to the dataset intializer:

.. code:: python

    dataset = Dataset(root='./data', transform=[Resize((64, 64)), ToTensor()])


Unfortunately, the text datasets in torchtext do not accept transformations in their init signatures; and, because they return vectorized text vectorize text, it's also not possible to include augmentation after reading in the data [#]_.


Available dataset classes
-------------------------

To facilitate the augmentation of text data in PyTorch pipelines, niacin includes a PyTorch compatibility module which provides three dataset classes capable of including text augmenting transformations. These are:

* niacin.text.compat.MemoryTextDataset: if the text data is already in memory but not on disk
* niacin.text.compat.FileTextDataset: if the text data are in a single file
* niacin.text.compat.DirectoryTextDataset: if the text data are in multiple files

For more information about each dataset type, see the API documentation.


Usage
-----

Now let's imagine that we have some simple data that we would like to use to train a sentiment classification model:

.. code:: python

    text = [
        "Oh my god! How could I not see this coming? My greed! I deserve this!",
        "Holy cats! Let's get outta here. I haven't learned a thing!"
    ]
    labels = [0, 1]


To use our functional transformations in this pipeline, we'll start by using ``functools.partial`` to freeze the probability value for a couple of transforms:

.. code:: python

    from functools import partial
    from niacin.text.en import add_whitespace, remove_whitespace

    transforms = [partial(add_whitespace, p=0.1), partial(remove_whitespace, p=0.1)]


Then, we can intialize a DataLoader class from PyTorch, then iterate over the data in our training loop like so:

.. code:: python

    from niacin.text.compat.pytorch import MemoryTextDataset
    from torch.utils.data import DataLoader

    dataset = MemoryTextDataset(data=text, labels=labels, transforms=transforms)
    loader = DataLoader(dataset)

    for epoch in range(3):
        print(epoch)
        for data, labels in loader:
            print(labels, data)


::

    0
    tensor([0]) tensor([[12, 24, 19,  2,  0,  3, 25, 28,  5, 15,  7, 11,  0,  0,  0,  0,  2,  3,
            0,  0, 29,  0,  2]])
    tensor([1]) tensor([[ 8, 14,  2, 10,  4, 27, 18, 26,  0,  0,  6,  3,  0,  0,  4, 29,  0,  0,
            13,  0,  0,  2]])
    1
    tensor([0]) tensor([[12, 24, 19,  2,  0,  3, 25, 28, 29,  0, 15,  7,  0,  2,  3, 17,  5,  2]])
    tensor([1]) tensor([[ 8, 14,  2, 10,  4, 27,  0,  0,  0,  0,  6,  0,  0,  4, 29,  0,  0,  0,
            13, 30,  2]])
    2
    tensor([0]) tensor([[12, 24, 19,  2,  9, 16,  3, 25, 28, 29,  0, 27, 15,  7,  0,  0, 20,  2,
            0,  5,  2]])
    tensor([1]) tensor([[ 8, 14,  2, 10,  4,  0,  0, 22,  6,  3, 21,  4, 29,  0, 30,  2]])



.. [#] c.f. `github.com/pytorch/text/issues/742 <https://github.com/pytorch/text/issues/742>`_