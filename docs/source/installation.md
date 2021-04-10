# Installation

The simplest way to install `textacy` is via `pip`:

```zsh
$ pip install textacy
```

or `conda`:

```zsh
$ conda install -c conda-forge textacy
```

If you prefer --- or are obliged --- you can download and unzip the source `tar.gz` from  PyPi, then install manually:

```zsh
$ python setup.py install
```

Dependencies
------------

Given the breadth of functionality, `textacy` depends on a number of other Python packages. Most of these are common components in the PyData stack (`numpy`, `scikit-learn`, etc.), but a few are more niche. One heavy dependency has been made optional.

Specifically: To use visualization functionality in `textacy.viz`, you'll need to have `matplotlib` installed. You can do so via `pip install textacy[viz]` or `pip install matplotlib`.

Downloading Data
----------------

For most uses of textacy, language-specific model data in spaCy is required. Fortunately, spaCy makes the process of getting this data easy; just follow [the instructions in their docs](https://spacy.io/docs/usage/models), which also includes a list of [currently-supported languages and their models](https://spacy.io/usage/models#languages).

**Note:** In previous versions of spaCy, users were able to link a specific model to a different name (e.g. "en_core_web_sm" => "en"), but this is no longer permitted. As such, `textacy` now requires users to fully specify which model to apply to a text, rather than leveraging automatic language identification to do it for them.

`textacy` itself features convenient access to several datasets comprised of thousands of text + metadata records, as well as a couple linguistic resources. Data can be downloaded via the `.download()` method on corresponding dataset/resource classes (see [Datasets and Resources](api_reference/datasets_resources) for details) or directly from the command line.

```zsh
$ python -m textacy download capitol_words
$ python -m textacy download depeche_mood
$ python -m textacy download lang_identifier --version 2.0
```

These commands download and save a compressed json file with ~11k speeches given by the main protagonists of the 2016 U.S. Presidential election, followed by a set of emotion lexicons in English and Italian with various word representations, and lastly a language identification model that works for 140 languages. For more information about particular datasets/resources use the `info` subcommand:

```zsh
$ python -m textacy info capitol_words
```
