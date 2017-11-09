import io
import os
import sys

import setuptools
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'cachetools>=2.0.0',
    'cytoolz>=0.8.0',
    'ftfy>=4.2.0,<5.0.0',
    'ijson>=2.3',
    'networkx>=1.11',
    'numpy>=1.9.0,<2.0.0',
    'pyemd>=0.3.0',
    'pyphen>=0.9.4',
    'python-levenshtein>=0.12.0',
    'requests>=2.10.0',
    'scipy>=0.17.0',
    'scikit-learn>=0.17.0',
    'spacy>=2.0.0',
    'tqdm>=4.11.1',
    'unidecode>=0.04.19',
    ]
EXTRAS_REQUIRE = {
    'viz': ['matplotlib>=1.5.0'],
    'lang': ['cld2-cffi>=0.1.4'],
    }
EXTRAS_REQUIRE['all'] = list({pkg for pkgs in EXTRAS_REQUIRE.values() for pkg in pkgs})

# as advised by https://hynek.me/articles/conditional-python-dependencies/
if int(setuptools.__version__.split('.')[0]) < 18:
    assert 'bdist_wheel' not in sys.argv
    if sys.version_info[0:2] == (2, 7):
        INSTALL_REQUIRES.append('backports.csv>=1.0.1')
else:
    EXTRAS_REQUIRE[':python_version=="2.7"'] = ['backports.csv>=1.0.1']


def read_file(fname, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding=encoding).read()


setup(
    name='textacy',
    version='0.4.1',
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Text Processing :: Linguistic',
        ],
    keywords='textacy, spacy, nlp, text processing, linguistics',

    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    )
