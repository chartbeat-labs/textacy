from contextlib import nullcontext as does_not_raise

import catalogue
import pytest
from spacy.tokens import Doc

import textacy


@pytest.fixture(scope="module")
def doc():
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice."
    )
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


class TestGetDocExtensions:
    @pytest.mark.parametrize(
        "name, context",
        [
            ("spacier", does_not_raise()),
            ("foo", pytest.raises(catalogue.RegistryError)),
            (None, pytest.raises(catalogue.RegistryError)),
        ],
    )
    def test_name(self, name, context):
        with context:
            _ = textacy.spacier.extensions.get_doc_extensions(name)

    def test_returns(self):
        result = textacy.spacier.extensions.get_doc_extensions("spacier")
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(val, dict) for val in result.values())
        assert all(
            isinstance(key, str) and callable(val)
            for value in result.values()
            for key, val in value.items()
        )


def test_extensions_exist(doc):
    for name in textacy.get_doc_extensions("spacier").keys():
        assert doc.has_extension(name)
        assert Doc.has_extension(name)


def test_set_remove_extensions(doc):
    textacy.remove_doc_extensions("spacier")
    for name in textacy.get_doc_extensions("spacier").keys():
        assert not doc.has_extension(name)
        assert not Doc.has_extension(name)
    textacy.set_doc_extensions("spacier")
    for name in textacy.get_doc_extensions("spacier").keys():
        assert doc.has_extension(name)
        assert Doc.has_extension(name)
