
Quickstart
==========


How do I install this?
----------------------

with ``pip``:

.. code:: sh

    pip install niacin

from source:

.. code:: sh

    git clone git@github.com:deniederhut/niacin.git && cd niacin && python setup.py install

How do I use this?
------------------

Functions in niacin are separated into submodules for specific data
types. Functions expose a similar API, with two input arguments: the
data to be transformed, and the probability of applying a specific
transformation.

enrichment:

.. code:: python

    from niacin import text
    data = "This is the song that never ends and it goes on and on my friends"
    print(text.add_misspelling(data, p=1.0))

.. code:: output

    This is teh song tath never ends adn it goes on anbd on my firends

negative sampling:

.. code:: python

    from niacin import text
    data = "This is the song that never ends and it goes on and on my friends"
    print(text.add_hypernyms(data, p=1.0))

.. code:: output

    This is the musical composition that never extremity and it exit on and on my person
