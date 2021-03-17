import os

import pytest

import textacy
from textacy.datasets.udhr import UDHR

DATASET = UDHR()


def _skipif():
    try:
        DATASET._check_data()
        return False
    except OSError:
        return True


pytestmark = pytest.mark.skipif(
    _skipif(),
    reason="UDHR dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = UDHR(data_dir=str(tmpdir))
    dataset.download()
    assert os.path.isfile(dataset._index_filepath)
    assert os.path.isdir(dataset._texts_dirpath)


def test_oserror(tmpdir):
    dataset = UDHR(data_dir=str(tmpdir))
    with pytest.raises(OSError):
        _ = list(dataset.texts())


def test_texts():
    texts = list(DATASET.texts(limit=3))
    assert len(texts) > 0
    for text in texts:
        assert isinstance(text, str)


def test_texts_limit():
    for limit in (1, 5, 10):
        assert sum(1 for _ in DATASET.texts(limit=limit)) == limit


def test_records():
    for text, meta in DATASET.records(limit=3):
        assert isinstance(text, str)
        assert isinstance(meta, dict)


def test_records_lang():
    langs = ({"en"}, {"en", "es"})
    for lang in langs:
        records = list(DATASET.records(lang=lang, limit=10))
        assert all(meta["lang"] in lang for _, meta in records)


def test_bad_filters():
    bad_filters = (
        {"lang": "xx"},
        {"lang": ["en", "un"]},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
    bad_filters = (
        {"lang": True},
        {"lang": textacy.load_spacy_lang("en_core_web_sm")},
    )
    for bad_filter in bad_filters:
        with pytest.raises(TypeError):
            list(DATASET.texts(**bad_filter))
