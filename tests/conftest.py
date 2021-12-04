import pytest

import textacy


@pytest.fixture(scope="session")
def lang_en():
    return textacy.load_spacy_lang("en_core_web_sm")


@pytest.fixture(scope="session")
def lang_es():
    return textacy.load_spacy_lang("es_core_news_sm")


@pytest.fixture(scope="session")
def text_en():
    return (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía "
        "was to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent "
        "that many things lacked names, and in order to indicate them it was necessary to point."
    )


@pytest.fixture(scope="session")
def text_es():
    return (
        "Muchos años después, frente al pelotón de fusilamiento, el coronel Aureliano Buendía "
        "había de recordar aquella tarde remota en que su padre lo llevó a conocer el hielo. "
        "Macondo era entonces una aldea de veinte casas de barro y cañabrava construidas a la orilla "
        "de un río de aguas diáfanas que se precipitaban por un lecho de piedras pulidas, "
        "blancas y enormes como huevos prehistóricos. El mundo era tan reciente, "
        "que muchas cosas carecían de nombre, y para mencionarlas había que señalarías con el dedo."
    )


@pytest.fixture(scope="session")
def doc_en(lang_en, text_en):
    meta = {"author": "Gabriel García Márquez", "title": "Cien años de soledad"}
    return textacy.make_spacy_doc((text_en, meta), lang=lang_en)


@pytest.fixture(scope="session")
def doc_es(lang_es, text_es):
    meta = {"autor": "Gabriel García Márquez", "título": "Cien años de soledad"}
    return textacy.make_spacy_doc((text_es, meta), lang=lang_es)


@pytest.fixture(scope="session")
def text_lines_en():
    return [
        "Mary had a little lamb, its fleece was white as snow.",
        "Everywhere that Mary went the lamb was sure to go.",
        "It followed her to school one day, which was against the rule.",
        "It made the children laugh and play to see a lamb at school.",
        "And so the teacher turned it out, but still it lingered near.",
        "It waited patiently about until Mary did appear.",
        "Why does the lamb love Mary so? The eager children cry.",
        "Mary loves the lamb, you know, the teacher did reply.",
    ]
