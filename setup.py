from codecs import open
from os import path
from setuptools import setup, find_packages


# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='textacy',
    version='0.1.1',

    description='Higher-level text processing, built on Spacy',
    long_description=long_description,
    url='https://github.com/chartbeat-labs/textacy',
    author='Burton DeWilde',
    author_email='burtondewilde@gmail.com',
    license='Apache',

    classifiers = [
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
    keywords = 'textacy, spacy, nlp, text processing, linguistics',

    packages=find_packages(),
    install_requires=[
        'cachetools',
        'cld2-cffi',
        'cytoolz',
        'ftfy',
        'fuzzywuzzy',
        'networkx',
        'nltk',
        'numpy>=1.8.0',
        'pandas',
        'pyphen',
        'scipy',
        'scikit-learn',
        'spacy>=0.100.0',
        'unidecode',
        ],
)
