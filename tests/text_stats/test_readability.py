from contextlib import nullcontext as does_not_raise

import pytest

import textacy
from textacy.text_stats import readability


@pytest.fixture(scope="module")
def en_doc_easy(lang_en):
    # https://simple.wikipedia.org/wiki/Climate_change
    text = (
        "Climate change means the climate of Earth changing. Climate change is now a big problem. "
        "Climate change this century and last century is sometimes called global warming, "
        "because the surface of the Earth is getting hotter. "
        "At times in the past, the temperature was much cooler, with the Ice Age ending about ten thousand years ago. "
        "Over very large time periods, climate change is caused by variations in the Earth's orbit around the Sun. "
        "The Earth has been much warmer and much cooler than it is today."
    )
    return textacy.make_spacy_doc(text, lang=lang_en)


@pytest.fixture(scope="module")
def en_doc_hard(lang_en):
    # https://en.wikipedia.org/wiki/Climate_change
    text = (
        "Contemporary climate change includes both the global warming caused by humans, "
        "and its impacts on Earth's weather patterns. There have been previous periods of climate change, "
        "but the current changes are more rapid than any known events in Earth's history. "
        "The main cause is the emission of greenhouse gases, mostly carbon dioxide (CO2) and methane. "
        "Burning fossil fuels for energy use creates most of these emissions. "
        "Agriculture, steel making, cement production, and forest loss are additional sources. "
        "Temperature rise is also affected by climate feedbacks such as the loss of sunlight-reflecting snow cover, "
        "and the release of carbon dioxide from drought-stricken forests. "
        "Collectively, these amplify global warming."
    )
    return textacy.make_spacy_doc(text, lang=lang_en)


@pytest.fixture(scope="module")
def es_doc(lang_es):
    # https://es.wikipedia.org/wiki/Calentamiento_global
    text = (
        "El calentamiento global es el aumento a largo plazo de la temperatura media del sistema climático "
        "de la Tierra. Es un aspecto primordial del cambio climático actual, "
        "demostrado por la medición directa de la temperatura y de varios efectos del calentamiento. "
        "Los términos calentamiento global y cambio climático a menudo se usan indistintamente, "
        "pero de forma más precisa calentamiento global es el incremento global en las temperaturas "
        "de superficie y su aumento proyectado causado predominantemente por actividades humanas (antrópico), "
        "mientras que cambio climático incluye tanto el calentamiento global como sus efectos en el clima. "
        "Si bien ha habido periodos prehistóricos de calentamiento global,​ varios de los cambios observados "
        "desde mediados del siglo XX no han tenido precedentes desde décadas a milenios."
    )
    return textacy.make_spacy_doc(text, lang=lang_es)


@pytest.mark.parametrize(
    "name, exp_easy, exp_hard, warns_lang",
    [
        ("automated_readability_index", 7.6, 11.8, False),
        ("automatic_arabic_readability_index", 1353.2, 1978.9, True),
        ("coleman_liau_index", 9.2, 13.9, False),
        ("flesch_kincaid_grade_level", 5.6, 8.6, False),
        ("flesch_reading_ease", 81.4, 61.9, False),
        ("gulpease_index", 63.4, 54.3, True),
        ("gunning_fog_index", 6.3, 10.3, False),
        ("lix", 35.1, 50.1, False),
        ("smog_index", 5.5, 10.3, False),
        ("wiener_sachtextformel", 2.1, 6.1, True),
    ],
)
def test_en_easy_hard(
    caplog, en_doc_easy, en_doc_hard, name, exp_easy, exp_hard, warns_lang
):
    func = getattr(readability, name)
    obs_easy = func(en_doc_easy)
    obs_hard = func(en_doc_hard)
    assert obs_easy == pytest.approx(exp_easy, abs=1.0)
    assert obs_hard == pytest.approx(exp_hard, abs=1.0)
    assert (
        "but this readability statistic is intended for use on" in caplog.text
    ) == warns_lang


@pytest.mark.parametrize(
    "name, exp, warns_lang",
    [
        ("flesch_reading_ease", 40.1, False),
        ("mu_legibility_index", 42.7, False),
        ("perspicuity_index", 35.5, False),
    ],
)
def test_es(caplog, es_doc, name, exp, warns_lang):
    func = getattr(readability, name)
    obs = func(es_doc)
    assert obs == pytest.approx(exp, abs=1.0)
    assert "but this readability statistic is intended for use on" not in caplog.text


@pytest.mark.parametrize(
    "lang, context",
    [
        (None, does_not_raise()),
        ("en", does_not_raise()),
        ("de", does_not_raise()),
        ("es", does_not_raise()),
        ("fr", does_not_raise()),
        ("it", does_not_raise()),
        ("nl", does_not_raise()),
        ("pt", does_not_raise()),
        ("ru", does_not_raise()),
        ("tr", pytest.raises(KeyError)),  # womp: pyphen lacks a turkish dictionary
        ("ja", pytest.raises(ValueError)),
        ("un", pytest.raises(ValueError)),
        ("zh", pytest.raises(ValueError)),
    ],
)
def test_flesch_reading_ease_lang(en_doc_easy, lang, context):
    with context:
        _ = readability.flesch_reading_ease(en_doc_easy, lang=lang)


def test_mu_legibility_index_doclen(es_doc):
    assert readability.mu_legibility_index(es_doc[:2].as_doc()) > 0.0
    assert readability.mu_legibility_index(es_doc[:1].as_doc()) == 0.0


@pytest.mark.parametrize(
    "variant, context",
    [
        (0, pytest.raises(ValueError)),
        (1, does_not_raise()),
        (2, does_not_raise()),
        (3, does_not_raise()),
        (4, does_not_raise()),
        (5, pytest.raises(ValueError)),
    ],
)
def test_wiener_sachtextformel_variant(en_doc_easy, variant, context):
    with context:
        _ = readability.wiener_sachtextformel(en_doc_easy, variant=variant)
