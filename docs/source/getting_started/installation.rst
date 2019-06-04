.. _installation:

Installation
============

The simplest way to install ``textacy`` is via ``pip``:

.. code-block:: console

    $ pip install textacy

or ``conda``:

.. code-block:: console

    $ conda install -c conda-forge textacy

If you prefer --- or are obliged --- to do things the hard way, you can download
and unzip the source ``tar.gz`` from  PyPi, then install manually:

.. code-block:: console

    $ python setup.py install

.. _installation_dependencies:

Dependencies
------------

Given the breadth of functionality, ``textacy`` depends on a number of other
Python packages. Most of these are typical components in the PyData stack,
but a few are certainly more niche. One heavy dependency has been made optional.

Specifically: To use visualization functionality, you'll need ``matplotlib`` installed;
you can do so via ``pip install textacy[viz]`` or ``pip install matplotlib``.

.. _installation_downloading-data:

Downloading Data
----------------

For most uses of textacy, language-specific model data in spaCy is required.
Fortunately, spaCy makes the process of getting this data easy and flexible;
just follow `the instructions in its docs <https://spacy.io/docs/usage/models>`_,
which also includes a list of `currently-supported languages and their models
<https://spacy.io/usage/models#section-available>`_.

**Note:** If you install specific versions of a given language's model data
(e.g. "en_core_web_sm" instead of just "en"), you'll want to create a shortcut link
to the corresponding standard two-letter form of the language so that it will
work as expected with textacy's automatic language identification. For example:

.. code-block:: console

    $ python -m spacy download en_core_web_sm
    $ python -m spacy link en_core_web_sm en

textacy itself features convenient access to several datasets comprised of
thousands of text + metadata records. Data can be downloaded via the ``.download()``
method on corresponding dataset classes (see :ref:`api-reference-datasets` for details)
or directly from the command line.

.. code-block:: console

    $ python -m textacy download capitol_words

will download and save a compressed json file with ~11k speeches given by the
main protagonists of the 2016 U.S. Presidential election (that had previously
served in the U.S. Congress). For more information, run

.. code-block:: console

    $ python -m textacy --help
