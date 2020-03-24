---
title: 'niacin: A Python package for text data enrichment'
tags:
  - data augmentation
  - natural language processing
  - machine learning
  - Python
authors:
  - name: Dillon Niederhut
    orcid: "0000-0002-9995-4378"
    affiliation: "1"
affiliations:
 - name: Novi Labs
   index: 1
date: 14 February 2020
bibliography: paper.bib

---

# Summary

A common component of frameworks for building robust and accurate learning models is a utility for performing sets of perturbations on the input data. In machine vision, for example, rotating, cropping, stretching, warping, and flipping training images have all been shown to increase model accuracy, and reduce certain kinds of overfitting [@chen2019invariance]. Popular machine vision libraries like PyTorch, Tensorflow, and FastAI all have built-in utilities for performing these transformations [@tensorflow2015;@NEURIPS2019_9015;@howard2020fastai].

In the domain of NLP, there is an analogous set of transformations that have been shown to increase model accuracy. Common transformations include adding superfluous whitespace, swapping an individual word for a synonym, replacing a word with a misspelled version, and round-tripping a sentence through a translation model into another language and back again [@zhang2015characterlevel;@coulombe2018text]. Some transforms, like misspellings or synonyms, attempt to directly model an unobserved but common change in the input features. Others, like adding a space in the middle of a word, act similarly to dropout for word-level models, in that they effectively remove the word as an input (e.g. the two halves of the word are unlikely to be present in the vocabulary).


There are also sets of transforms whose aim is to make models more robust in the face of adversarial inputs. These transforms include replacing letters with symbols that are visually similar, adding nuisance characters around words, adding random characters at the end of the document, and appending the word "love" to the end of a sentence [@grndahl2018need]. Adding nuisance characters around words, or replacing characters with symbols, disrupt the behavior of vocabulary-based models, in that they will not know how to encode an input like "sch00l", although it is visually clear to an experienced reader of English that this is the word "school". Adding words with very positive or very negative connotations have been shown to be an effective way to disrupt sentiment classifiers, which often rely on counting the number of times such words appear.

Here, we introduce niacin, a python library for performing a large set of these commonly-used text transformations on a given input, with some probability. For example, a text augmentation technique used in technologies like word2vec [@mikolov2013distributed] and Glove [@pennington2014glove] is to produce negative examples (bad or wrong inputs) by replacing words with ones that are too specific for the context of the rest of the sentence. In niacin, you can use a function called `add_hyponyms` for this, which, at a replacement probability of 1.0, produces outputs like this:

```
>>> add_hyponyms("this is the song that never ends and it goes on
and on my friends", p=1.0)

'this is the aria that never closure and it shove off on and on my confidant'
```

There are also functions for replacing words with common misspellings:

```
>>> add_misspelling("There are also functions for replacing words with
their common misspellings", 1.0)
'There are aslo functions for replacing words withh thier common misspellings'
```

replacing contractions:

```
>>> add_contractions('You are rad', 1.0)
"you're rad"
```

and substituting numbers and symbols for letters they resemble:

```
>>> add_leet("Hello, how are you?", 1.0)
'H3110, h0w r u?'
```

When possible, function documentation links back to the paper which originally suggested or implemented the transformation, should a user want more information about how apply the function in practice. For example, replacing every single character in your text input will render it useless, but there is emprical work suggesting that no more than a 10% probability of replacement should be used. Data sources (principally WordNet [@wordnet], but also Wikipedia) are always cited.

There are a small number of libraries that contain text augmentation functions, for example [noisemix](https://github.com/noisemix/noisemix) and [EDA](https://github.com/jasonwei20/eda_nlp) [@wei2019eda]. However, many of these are published only as a part of a conference paper, and are not actively maintained. EDA, for example, contains the warning "The code is not documented". They also tend to contain a subset of the common kinds of text transformations. EDA contains four functions, two of which are varities of synonym replacement. Noisemix is more feature complete, and contains a function that adds typos to words. There is not an exact one-to-one correspondence between all the capabilities of each library, but here is a table with a rough alignment:


transformation       | niacin | noisemix | eda
---------------------|--------|----------|----
synonym replacement  | x      | x        | x
hypernym replacement | x      |          |
hyponym replacement  | x      |          |
swap word order      |        | x        | x
swap character order |        | x        |
remove articles      | x      |          |
contract/expand word | x      |          |
remove punctuation   |        | x        |
add/delete space     | x      | x        |
add/delete letter    |        | x        |
add typo             | x      | x        |
add misspelling      | x      |          |
wrap in parens       | x      |          |
add applause emoji   | x      |          |
add 'love'           | x      |          |
add random characters| x      |          |


It is our hope that having a collection of already-implemented transformations with a uniform interface will make it easy for researchers to include them in their own data processing pipelines.

# References
