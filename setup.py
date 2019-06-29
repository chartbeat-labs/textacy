import io
import os
import sys

import setuptools
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    "cachetools>=2.0.0",
    "cytoolz>=0.8.0",
    "jellyfish>=0.6.0,<0.7.0",
    "joblib>=0.13.0",
    "networkx>=1.11",
    "numpy>=1.9.0,<2.0.0",
    "pyemd>=0.3.0",
    "pyphen>=0.9.4",
    "requests>=2.10.0",
    "scipy>=0.17.0",
    "scikit-learn>=0.18.0,<0.21.0",
    "spacy>=2.0.12",
    "srsly>=0.0.5",
    "tqdm>=4.11.1",
]
EXTRAS_REQUIRE = {"viz": ["matplotlib>=1.5.0"]}
EXTRAS_REQUIRE["all"] = list({pkg for pkgs in EXTRAS_REQUIRE.values() for pkg in pkgs})

# as advised by https://hynek.me/articles/conditional-python-dependencies/
if int(setuptools.__version__.split(".")[0]) < 18:
    assert "bdist_wheel" not in sys.argv
    if sys.version_info[0:2] == (2, 7):
        INSTALL_REQUIRES.append("backports.csv>=1.0.1")
else:
    EXTRAS_REQUIRE[':python_version=="2.7"'] = ["backports.csv>=1.0.1"]


def read_file(fname, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), fname)
    with io.open(path, encoding=encoding) as f:
        data = f.read()
    return data


about = {}
root_path = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(root_path, "textacy", "about.py")) as f:
    exec(f.read(), about)


setup(
    name="textacy",
    version=about["__version__"],
    description=about["__description__"],
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url=about["__url__"],
    download_url=about["__download_url__"],
    maintainer=about["__maintainer__"],
    maintainer_email=about["__maintainer_email__"],
    license=about["__license__"],
    python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="textacy, spacy, nlp, text processing, linguistics",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
