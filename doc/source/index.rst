.. niacin documentation master file, created by
   sphinx-quickstart on Tue Nov 27 12:18:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to niacin's documentation!
==================================

A Python library for replacing the missing variation in your data.


Why should I use ``niacin``?
----------------------------

Data collected for model training necessarily undersamples the likely
variance in the input space. This library is a collection of tools for
inserting typical kinds of perturbations to better approximate population
variance; and, for creating similar-but-incorrect examples to aid in
reducing the total size of the hypothesis space. These are commonly known as
ENRICHMENT and NEGATIVE SAMPLING, respectively.


.. toctree::
   :maxdepth: 3

   quickstart
   user_guide/index
   niacin


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
