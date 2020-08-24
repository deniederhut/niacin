.. _with-randaugment:

using RandAugment with niacin
===============================

RandAugment (randaugment_) is an algorithm for applying a variety of combinations of augmentations to training data, and experimentally appears to produce results on par with more complicated data augmentation policies like AutoAugment (autoaugment_).

In the RandAugment algorithm, there is a predefined set of ``k`` augmentation functions. For each training sample, a random subset of size ``n<<k`` transformation functions are drawn, and applied to the individual sample with a strength or magnitude ``m``. Here, k, n, and m are hyperparameters to the augmentation policy.

Usage
-----

To use RandAugment with enrichment functions from niacin, start by creating an instance of the RandAugment class, and provide a set of transformation functions to be considered. You can set n and m directly, or search over them via a hyperparameter tuning algorithm.


.. code:: python

    from niacin.augment import RandAugment
    from niacin.text import en

    augmentor = RandAugment([
        en.add_synonyms,
        en.add_hyponyms,
        en.add_misspelling,
        en.swap_words,
        en.add_contractions,
        en.add_whitespace,
    ], n=2, m=15, shuffle=False)


You can now use use this object directly, to augment input data by hand:


.. code:: python

    text = [
        "No reading or writing makes a savage of men",
        "They were praying for jail, but I mastered the pen",
    ]
    for data in text:
        for tx in augmentor:
            data = tx(data)
        print(data)

returns

::

    No reading or wr it ing make a savage of  men
    They were praying for jail, but I mastered the ballpoint


With PyTorch Datasets
------------------------


If you are using the PyTorch Dataset classes defined in niacin [#]_, you can give the augmentor to the Dataset class, and have it apply the transformations on the fly when it retrieves data:


.. code:: python

    from niacin.text.compat.pytorch import MemoryTextDataset
    from torch.utils.data import DataLoader

    text = [
        "Tell them how we are funding all of these kids to go to college",
        "Tell them how we are ceasing all these wars and stopping violence",
    ]

    dataset = MemoryTextDataset(data=text, labels=[1, 1], transforms=augmentor)
    loader = DataLoader(dataset)

    for epoch in range(3):
        print(epoch)
        for data, labels in loader:
            print(labels, data)

returns

::

    0
    tensor([1]) tensor([[ 0,  0,  6,  0,  0,  9,  0,  0,  0,  0,  0,  0,  0, 16,  7,  0,  8, 14,
            0,  0,  0,  0]])
    tensor([1]) tensor([[ 2,  6,  5,  9,  4, 11,  3,  7,  0, 10, 17, 18]])
    1
    tensor([1]) tensor([[ 2,  6,  5,  9,  4, 13,  3, 16,  7,  0,  8, 14,  8, 12]])
    tensor([1]) tensor([[ 2,  6,  9,  5,  4, 11,  3,  7, 19, 10, 17, 18]])
    2
    tensor([1]) tensor([[ 2,  6,  5,  9,  4, 13,  3, 16,  0,  0,  8, 14,  8, 12]])
    tensor([1]) tensor([[ 2,  6,  5,  9,  0, 11,  3,  7,  0, 10, 18, 17]])


.. [#] :ref:`with-pytorch`

.. _autoaugment: https://arxiv.org/abs/1805.09501
.. _randaugment: https://arxiv.org/abs/1909.13719