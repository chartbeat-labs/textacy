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
def record_en(text_en):
    meta = {"author": "Gabriel García Márquez", "title": "One Hundred Years of Solitude"}
    return (text_en, meta)


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
def record_es(text_es):
    meta = {"autor": "Gabriel García Márquez", "título": "Cien años de soledad"}
    return (text_es, meta)


@pytest.fixture(scope="session")
def doc_en(lang_en, record_en):
    return textacy.make_spacy_doc(record_en, lang=lang_en)


@pytest.fixture(scope="session")
def doc_es(lang_es, record_es):
    return textacy.make_spacy_doc(record_es, lang=lang_es)


@pytest.fixture(scope="session")
def texts_en():
    return (
        "Contemporary climate change includes both the global warming caused by humans, and its impacts on Earth's weather patterns. There have been previous periods of climate change, but the current changes are more rapid than any known events in Earth's history.",
        "Biodiversity loss includes the worldwide extinction of different species, as well as the local reduction or loss of species in a certain habitat, resulting in a loss of biological diversity.",
        "Pollution is the introduction of contaminants into the natural environment that cause adverse change.[1] Pollution can take the form of any substance (solid, liquid, or gas) or energy (such as radioactivity, heat, sound, or light).",
        "Air pollution is the presence of substances in the atmosphere that are harmful to the health of humans and other living beings, or cause damage to the climate or to materials.",
        "The conservation movement, also known as nature conservation, is a political, environmental, and social movement that seeks to manage and protect natural resources, including animal, fungus, and plant species as well as their habitat for the future. Conservationists are concerned with leaving the environment in a better state than the condition they found it in.",
    )


@pytest.fixture(scope="session")
def records_en(texts_en):
    metas = (
        {
            "title": "Climate change",
            "categories": ("Anthropocene", "Environmental issues"),
            "num_sections": 12,
        },
        {
            "title": "Biodiversity loss",
            "categories": ("Environmental issues", "Extinction"),
            "num_sections": 11,
        },
        {
            "title": "Pollution",
            "categories": ("Pollution", "Environmental toxicology"),
            "num_sections": 12,
        },
        {
            "title": "Air Pollution",
            "categories": ("Pollution", "Climate forcing"),
            "num_sections": 19,
        },
        {
            "title": "Conservation movement",
            "categories": ("Environmental conservation", "Environmental ethics"),
            "num_sections": 7,
        },
    )
    return tuple((text, meta) for text, meta in zip(texts_en, metas))


@pytest.fixture(scope="session")
def docs_en(lang_en, records_en):
    return tuple(textacy.make_spacy_doc(record, lang_en) for record in records_en)


@pytest.fixture(scope="session")
def texts_short_en():
    return (
        "Mary had a little lamb, its fleece was white as snow.",
        "Everywhere that Mary went the lamb was sure to go.",
        "It followed her to school one day, which was against the rule.",
        "It made the children laugh and play to see a lamb at school.",
        "And so the teacher turned it out, but still it lingered near.",
        "It waited patiently about until Mary did appear.",
        "Why does the lamb love Mary so? The eager children cry.",
        "Mary loves the lamb, you know, the teacher did reply.",
    )


@pytest.fixture(scope="session")
def text_long_en():
    return (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice. At that time Macondo was a village of twenty adobe houses, built on the bank of a river of clear water that ran along a bed of polished stones, which were white and enormous, like prehistoric eggs. The world was so recent that many things lacked names, and in order to indicate them it was necessary to point. Every year during the month of March a family of ragged gypsies would set up their tents near the village, and with a great uproar of pipes and kettledrums they would display new inventions. First they brought the magnet. A heavy gypsy with an untamed beard and sparrow hands, who introduced himself as Melquíades, put on a bold public demonstration of what he himself called the eighth wonder of the learned alchemists of Macedonia. He went from house to house dragging two metal ingots and everybody was amazed to see pots, pans, tongs and braziers tumble down from their places and beams creak from the desperation of nails and screws trying to emerge, and even objects that had been lost for a long time appeared from where they had been searched for most and went dragging along in turbulent confusion behind Melquíades' magical irons. 'Things have a life of their own,' the gypsy proclaimed with a harsh accent. 'It's simply a matter of waking up their souls.' José Arcadio Buendía, whose unbridled imagination always went beyond the genius of nature and even beyond miracles and magic, thought that it would be possible to make use of that useless invention to extract gold from the bowels of the earth. Melquíades, who was an honest man, warned him: 'It won't work for that.' But José Arcadio Buendía at that time did not believe in the honesty of gypsies, so he traded his mule and a pair of goats for the two magnetized ingots. Úrsula Iguarán, his wife, who relied on those animals to increase their poor domestic holdings, was unable to dissuade him. 'Very soon we'll have gold enough and more to pave the floors of the house,' her husband replied. For several months he worked hard to demonstrate the truth of his idea. He explored every inch of the region, even the riverbed, dragging the two iron ingots along and reciting Melquíades' incantation aloud. The only thing he succeeded in doing was to unearth a suit of fifteenth-century armour which had all of its pieces soldered together with rust and inside of which there was the hollow resonance of an enormous stone-filled gourd. When José Arcadio Buendía and the four men of his expedition managed to take the armour apart, they found inside a calcified skeleton with a copper locket containing a woman's hair around its neck.\n\n"
        "In March the gypsies returned. This time they brought a telescope and a magnifying glass the size of a drum, which they exhibited as the latest discovery of the Jews of Amsterdam. They placed a gypsy woman at one end of the village and set up the telescope at the entrance to the tent. For the price of five reales, people could look into the telescope and see the gypsy woman an arm's length away. 'Science has eliminated distance,' Melquíades proclaimed. 'In a short time, man will be able to see what is happening in any place in the world without leaving his own house.' A burning noonday sun brought out a startling demonstration with the gigantic magnifying glass: they put a pile of dry hay in the middle of the street and set it on fire by concentrating the sun's rays. José Arcadio Buendía, who had still not been consoled for the failure of his magnets, conceived the idea of using that invention as a weapon of war. Again Melquíades tried to dissuade him, but he finally accepted the two magnetized ingots and three colonial coins in exchange for the magnifying glass. Úrsula wept in consternation. That money was from a chest of gold coins that her father had put together over an entire life of privation and that she had buried underneath her bed in hopes of a proper occasion to make use of it. José Arcadio Buendía made no attempt to console her, completely absorbed in his tactical experiments with the abnegation of a scientist and even at the risk of his own life. In an attempt to show the effects of the glass on enemy troops, he exposed himself to the concentration of the sun's rays and suffered burns which turned into sores that took a long time to heal. Over the protests of his wife, who was alarmed at such a dangerous invention, at one point he was ready to set the house on fire. He would spend hours on end in his room, calculating the strategic possibilities of his novel weapon until he succeeded in putting together a manual of startling instructional clarity and an irresistible power of conviction. He sent it to the government, accompanied by numerous descriptions of his experiments and several pages of explanatory sketches, by a messenger who crossed the mountains, got lost in measureless swamps, forded stormy rivers, and was on the point of perishing under the lash of despair, plague, and wild beasts until he found a route that joined the one used by the mules that carried the mail. In spite of the fact that a trip to the capital was little less than impossible at that time, José Arcadio Buendía promised to undertake it as soon as the government ordered him to so that he could put on some practical demonstrations of his invention for the military authorities and could train them himself in the complicated art of solar war. For several years he waited for an answer. Finally, tired of waiting, he bemoaned to Melquíades the failure of his project and the gypsy then gave him a convincing proof of his honesty: he gave him back the doubloons in exchange for the magnifying glass, and he left him in addition some Portugues maps and several instruments of navigation. In his own handwriting he set down a concise synthesis of the studies by Monk Hermann, which he left José Arcadio so that he would be able to make use of the astrolabe, the compass, and the sextant. José Arcadio Buendía spent the long months of the rainy season shut up in a small room that he had built in the rear of the house so that no one would disturb his experiments. Having completely abandoned his domestic obligations, he spent entire nights in the courtyard watching the course of the stars and he almost contracted sunstroke from trying to establish an exact method to ascertain noon. When he became an expert in the use and manipulation of his instruments, he conceived a notion of space that allowed him to navigate across unknown seas, to visit uninhabited territories, and to establish relations with splendid beings without having to leave his study. That was the period in which he acquired the habit of talking to himself, of walking through the house without paying attention to anyone, as Úrsula and the children broke their backs in the garden, growing banana and caladium, cassava and yams, ahuyama roots and eggplants. Suddenly, without warning, his feverish activity was interrupted and was replaced by a kind of fascination. He spent several days as if he were bewitched, softly repeating to himself a string of fearful conjectures without giving credit to his own understanding. Finally, one Tuesday in December, at lunchtime, all at once he released the whole weight of his torment. The children would remember for the rest of their lives the august solemnity with which their father, devastated by his prolonged vigil and by the wrath of his imagination, revealed his discovery to them:\n\n"
        "'The earth is round, like an orange.'"
    )


@pytest.fixture(scope="session")
def doc_long_en(lang_en, text_long_en):
    return textacy.make_spacy_doc(text_long_en, lang=lang_en)
