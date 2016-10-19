import io
import os

from setuptools import setup, find_packages


def read_file(fname, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding=encoding).read()


setup(
    name='textacy',
    version='0.3.1',
    description='Higher-level text processing, built on spaCy',
    long_description=read_file('README.rst'),

    url='https://github.com/chartbeat-labs/textacy',
    download_url='https://pypi.python.org/pypi/textacy',

    author='Burton DeWilde',
    author_email='burtdewilde@gmail.com',
    license='Apache',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Topic :: Text Processing :: Linguistic',
        ],
    keywords='textacy, spacy, nlp, text processing, linguistics',

    packages=find_packages(),
    install_requires=[
        'backports.csv>=1.0.1',
        'cachetools>=2.0.0',
        'cld2-cffi>=0.1.4',
        'cytoolz>=0.8.0',
        'ftfy>=4.2.0',
        'fuzzywuzzy>=0.12.0',
        'gensim>=0.13.2',
        'ijson>=2.3',
        'matplotlib>=1.5.0',
        'networkx>=1.11',
        'numpy>=1.8.0',
        'pyemd>=0.3.0',
        'pyphen>=0.9.4',
        'python-levenshtein>=0.12.0',
        'requests>=2.10.0',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.0',
        'spacy>=1.0.1',
        'unidecode>=0.04.19',
        ],
    )
