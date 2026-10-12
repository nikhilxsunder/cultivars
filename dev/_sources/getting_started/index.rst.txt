.. meta::
   :description: Install cultivars, fit a first model, and learn the model-fit-result pattern every estimator in the library follows.

Getting started
===============

cultivars is built around one pattern. A *model* holds the data and the
specification, ``fit()`` estimates it, and the *result* it returns is a
frozen record of everything the estimate is: parameters, residuals,
diagnostics, and a summary that states what the numbers assume. Learn
the pattern once and every estimator in the library reads the same way,
from an ARIMA to a sign-restricted structural VAR.

The three pages below take about twenty minutes in order. Readers who
already know the econometrics can start at the quickstart; readers who
want to know why the library is shaped the way it is can start at the
concepts.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Supported Pythons, optional extras, pinning a pre-release, and a
      development checkout.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Test a series for a unit root, fit and compare ARIMA models, then
      a VAR: order selection, forecasts, causality, and impulse
      responses.

   .. grid-item-card:: Core concepts
      :link: concepts
      :link-type: doc

      Models, results, summaries, hypothesis tests, identification as a
      second stage, data conventions, and the error hierarchy.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   quickstart
   concepts

Where next
----------

After these pages, the :doc:`examples <../auto_examples/index>` run
complete research workflows on FRED data, and the
:doc:`API reference <../api/index>` documents every public class with
its mathematics, references, and a runnable example.
