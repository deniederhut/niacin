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

In the domain of natural language processing (NLP), there is an analogous set of transformations that have been shown to increase model accuracy. Common transformations include adding superfluous whitespace, swapping an individual word for a synonym, replacing a word with a misspelled version, and round-tripping a sentence through a translation model into another language and back again [@zhang2015characterlevel;@coulombe2018text]. Some transforms, like misspellings or synonyms, attempt to directly model an unobserved but common change in the input features. Others, like adding a space in the middle of a word, act similarly to dropout for word-level models, in that they effectively remove the word as an input (e.g., the two halves of the word are unlikely to be present in the vocabulary).

For specific tasks in NLP, principally in language modeling, model improvement can be effected with incorrect examples of input data, so that the model learns not to produce them. This practice is called negative sampling, and was used in the development of both word2vec [@mikolov2013distributed] and GLoVE [@pennington2014glove]. This can be done with words sampled from the whole vocabulary, or in a more targeted fashion by including related, but inappropriate tokens. For example, in English, "I ate pizza for lunch" is a valid sentence. Replacing "pizza" and "lunch" with a higher level category (these are called hypernyms), produces -- "I ate _dish_ for _meal_" an output that is technically well-formed, but violates the Gricean expectation of utility.

There are also sets of transforms whose aim is to make models more robust in the face of adversarial inputs. These transforms include replacing letters with symbols that are visually similar, adding nuisance characters around words, adding random characters at the end of the document, and appending the word "love" to the end of a sentence [@grndahl2018need]. Adding nuisance characters, or replacing characters with symbols, disrupts the behavior of vocabulary-based models, in that they will not know how to encode an input like "sch00l", although it is visually clear to an experienced reader of English that this is the word "school". Adding words with very positive or very negative connotations have been shown to be an effective way to disrupt sentiment classifiers, which often rely on counting the number of times such words appear.

Here, we introduce niacin, a python library for performing a large set of these commonly-used text transformations on a given input, with some probability. For example, a commonly used text augmentation technique is to replace individual words in a sentence with ones that are closely related. The idea here is that the sentences "I am unhappy" and "I am sad" convey a similar semantic value, even though the individual words are different. In niacin, you can use a function called `add_synonyms` for this, which, at a replacement probability of 1.0:

```
>>> add_synonyms("this is the song that never ends and it goes on
and on my friends", p=1.0)
```

produces outputs like this:

> this is the _vocal_ that _ne'er terminate_ and it _go away_ on and on my friend

There are also functions for replacing words with common misspellings:

```
>>> add_misspelling("There are also functions for replacing words with
their common misspellings", 1.0)
```

> There are _aslo_ functions for replacing words _withh_ _thier_ common misspellings


replacing contractions:

```
>>> add_contractions('You are rad', 1.0)
```

> _you're_ rad

and substituting numbers and symbols for letters they resemble:

```
>>> add_leet("Hello, how are you?", 1.0)
```

> _H3110_, _h0w_ _r_ _u_?

When possible, function documentation links back to the paper which originally suggested or implemented the transformation, should a user want more information about how to apply the function in practice. For example, replacing every single character in your text input will render it useless, but there is empirical work suggesting that no more than a 10% probability of replacement should be used. Data sources (principally WordNet [@wordnet], but also Wikipedia) are always cited.

In 2018, when niacin was first published online, there were a small number of libraries that contain text augmentation functions, for example [noisemix](https://github.com/noisemix/noisemix) and [EDA](https://github.com/jasonwei20/eda_nlp) [@wei2019eda]. In general, these were published only as a part of a conference paper, and were not actively maintained. EDA, for example, contains the warning "The code is not documented". They also tended to be limited to the scope of the transformations included in the initial paper. There is not an exact one-to-one correspondence between all the capabilities of these libraries, but this table shows an approximation of comparing features across them:

transformation       | niacin | noisemix | eda
---------------------|--------|----------|----
synonym replacement  | x      | x        | x
hypernym replacement | x      |          |
hyponym replacement  | x      |          |
swap word order      | x      | x        | x
swap character order | x      | x        |
remove articles      | x      |          |
contract/expand word | x      |          |
remove punctuation   | x      | x        |
add/delete space     | x      | x        |
add/delete letter    | x      | x        |
add typo             | x      | x        |
add misspelling      | x      |          |
wrap in parens       | x      |          |
add applause emoji   | x      |          |
add 'love'           | x      |          |
add random characters| x      |          |
add backtranslation  | x      |          |

More recently, other software tools have been published that cover some parts of niacin's functionality, and extend further in other areas. `nlpaug`, for example, includes large language models like BERT that can perform synonym replacement in a way that is context-dependent -- something that wordnet cannot do [@ma2019nlpaug]. However, it does not include some of the character-based transformations, nor the adversarial ones. `TextAttack` implements several recently published methods in model-based adversarial attacks on language models, in both the white-box and black-box domain, and includes a small number of augmentation techniques in addition [@Morris2020TextAttack].

It is our hope that having a collection of already-implemented transformations with a uniform interface will make it easy for researchers to include them in their own data processing pipelines.

# References
