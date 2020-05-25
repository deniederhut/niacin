
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

If you have installed ``niacin`` from source, you can run the test suite to verify that
everything is working properly. We use `pytest <https://docs.pytest.org/en/latest/>`_,
which you will first need to install:

.. code:: sh

    pip install pytest


then you can run the library's tests with

.. code:: sh

    pytest -m 'not slow'


if you would like to see the coverage report, you can do so with `pytest-cov`
like so:

.. code:: sh

    pip install pytest-cov
    pytest -m 'not slow' --cov=niacin && coverage html


How can I install the optional dependencies?
--------------------------------------------

If you want to use the backtranslate functionality, niacin will need pytorch and some other
libraries. These can be installed as extras with:

.. code:: sh

    pip install niacin[backtranslate]

If you are on macos, this might fail with a warning about your version of gcc:

.. code:: output

    Your compiler (g++) is not compatible with the compiler Pytorch was
    built with for this platform, which is clang++ on darwin.

You can avoid this error by executing the following:

.. code:: sh
    CFLAGS='-stdlib=libc++' pip install niacin[backtranslate]


How do I use this?
------------------

Functions in niacin are separated into submodules for specific data
types. Functions expose a similar API, with two input arguments: the
data to be transformed, and the probability of applying a specific
transformation.

enrichment:

.. code:: python

    from niacin.text import en
    data = "This is the song that never ends and it goes on and on my friends"
    print(en.add_misspelling(data, p=1.0))

.. code:: output

    This is teh song tath never ends adn it goes on anbd on my firends

negative sampling:

.. code:: python

    from niacin.text import en
    data = "This is the song that never ends and it goes on and on my friends"
    print(en.add_hypernyms(data, p=1.0))

.. code:: output

    This is the musical composition that never extremity and it exit on and on my person
