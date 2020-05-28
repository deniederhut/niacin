english text usage
==================

The text transformation functions in niacin fall into three high-level categories, which have different intended uses. These are:

1. ecologically plausible transformations that do not change the semantic content (`Enrichment`_)
2. ecologically plausible transformations that modify the semantic content (`Negative Sampling`_)
3. ecologically improbable transformations that do not change the semantic content (`Adversarial Inputs`_)

We'll provide more details about each of these below.


Enrichment
----------

In data enrichment, we attempt to create more valid samples of data than we have actual observations. We might do this because we have a high variance model and not enough data to fit it well, or we might do it to make our model more robust to the kinds of changes it should expect to see in the wild. As an example, let's imagine that we are trying to train a model to categorize sentences by topic. We might have a dataset like this:

.. code:: python

    text = "A four-alarm fire destroys a warehouse at Pier 45 at Fisherman's Wharf in San Francisco, California."

As it stands, a model might not infer that "esrquake" was intended to be "earthquake". Since earthquake is an important word in determine that this sentence is about a disaster, handling small changes will help our model generalize to situations where documents can have words spelled incorrectly. To simulate this scenario with niacin, we can apply the ``add_fat_thumbs`` function to the column of text data:

.. code:: python

    from niacin.text import en
    en.add_fat_thumbs(text, p=0.05)

::

    "A 5.9 magnltude esrthquake ztrikes 60 miles (97 km) west of Wellihgton, New Zeqland. No injuries are eeported."

We might also want the model to generalize across closely related words. For example, "flame" is close to "fire". We can force the model to learn this by replacing words with their synonyms:

.. code:: python

    en.add_synonyms(text)

::

    "type A four-alarm flame destroys a storage warehouse at Pier 45 at Fisherman's Wharf in San Francisco, California."

.. note::

    You can also chain together transformations in sequence, being careful to apply them in the right order. In general, you want sentence-level transformations first, then word-level, then character-level, because lower-level modifications might disrupt the ability of the higher-level ones to, for example, discover word boundaries, or find the lexemes needed for synonym search.

    .. code:: python

        en.add_fat_thumbs(en.add_synonyms(text, p=0.5), p=0.05)

    ::

        "A four-alarm flame destroys a storage warehouse at Pier 45 at Fiaherman's Wharf in San Francisco, California."

The enrichment transformations include:

* add_backtranslation
* add_characters
* add_contractions
* add_fat_thumbs
* add_synonyms
* add_whitespace
* remove_articles
* remove_characters
* remove_contractions
* remove_punctuation
* remove_whitespace
* swap_chars
* swap_words

.. warning::

    In general, applying too many transformations (or each transformation with too high a probability value) will create text that is unrecognizable, and will not help your model learn.

    .. code:: python

        en.add_fat_thumbs(text, p=1.0)

    ::

        "A gkie-q/zth roff csxgf;hq s axefuiycf sf Pkwd 45 qf Fuzndghsj'c Wnstb kj Sqh Fdwbfuevp, Cqkucptmos."


Negative Sampling
-----------------

In negative sampling, we are attempting to teach our model bad or wrong examples. This is commonly used in generating word embeddings, as a way to not only teach the model which words appear in similar contexts, but also which ones should not. Let's imagine we have the following input:

.. code:: python

    text = "A 5.9 magnitude earthquake strikes 60 miles (97 km) west of Wellington, New Zealand. No injuries are reported."

and we want our model to learn to disassociate some words that should not be considered to be related to earthquakes. We can take the second sentence in the input and transform with a method that does not preserve the semantics of the input, like a hypernym.

.. code:: python

    from niacin.text import en
    en.add_hypernyms(text[-25:], p=1.0)

::

    'No ill health area unit reported.'

The negative sampling transformations include:

* add_hypernyms
* add hyponyms

Adversarial Inputs
------------------

An adversarial input is one which is intended to cause model failures -- predicting the wrong label for an input -- and is commonly used to get around content filtering algorithms like spam detection. Let's say for example that our model is a sentiment classifier, and we have an input that looks like this:

.. code:: python

    from niacin.text import en
    text = "A Pakistan International Airlines passenger aircraft crashes in Karachi, killing ninety-seven people."

A good model would hopefully categorize this with a negative sentiment, since this is a sad event. Someone trying to fool a model into classifying this as a positive news story might do something like the following:

.. code:: python

    en.add_love(text, 1.0)

::

    'A Pakistan International Airlines passenger aircraft crashes in Karachi, killing ninety-seven people. love'

This is surprisingly effective, because real examples of sentiment do not tend to include "love" in describing otherwise bad or sad events. An adversary might also want the sentence unclassifiable, or classified as neutral. For word-based models, an effective strategy includes modifying the spelling of words, manipulating the whitespace around words, or including other tokens. For example, wrapping every word with parentheses can cause none of the tokens to be included in the model's vocabulary, and it will be unable to generate an appropriate output:

.. code:: python

    en.add_parens(text, 1.0)

::

    '(((A))) (((Pakistan))) (((International))) (((Airlines))) (((passenger))) (((aircraft))) (((crashes))) (((in))) (((Karachi,))) (((killing))) (((ninety-seven))) (((people.)))'

Including these kinds of transforms in the input data to a model can make them more robust to these kinds of transformations, given the model has the capacity to represent them.

The adversarial transformations include:

* add_applause
* add_bytes
* add_leet
* add_love
* add_parens