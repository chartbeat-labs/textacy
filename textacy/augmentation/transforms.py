import collections
import itertools
import random

# from cytoolz import dicttoolz
from spacy.tokens import Doc, Span

from .. import resources
# from ..doc import make_spacy_doc


rs = resources.ConceptNet()

AugTok = collections.namedtuple("AugTok", ["text", "ws", "pos", "syns", "is_punct"])


# TODO: replace this with something better, and maybe move it :)
# def apply(
#     doc,
#     n_replacements=1,
#     n_insertions=1,
#     n_swaps=1,
#     delete_prob=0.05,
#     shuffle_sents=True,
# ):
#     """
#     Apply a variety of transformations to the sentences in ``doc`` to generate
#     a similar-but-different document, suitable for improving performance
#     on text classification tasks.

#     Args:
#         doc (:class:`spacy.tokens.Doc`): Text document to be augmented through
#             a variety of transformations.
#         n_replacements (int): Maximum number of items to replace with synonyms,
#             per sentence.
#         n_insertions (int): Maximum number of times to insert synonyms, per sentence.
#         n_swaps (int): Maximum number of times to swap items, per sentence.
#         delete_prob (float): Probability that any given item is deleted.
#         shuffle_sents (bool): If True, shuffle the order of sentences in ``doc``;
#             otherwise, leave sentence order unchanged.

#     Returns:
#         :class:`spacy.tokens.Doc`: New, transformed document generated from ``doc``.

#     References:
#         Wei, Jason W., and Kai Zou. "Eda: Easy data augmentation techniques
#         for boosting performance on text classification tasks."
#         arXiv preprint arXiv:1901.11196 (2019).
#     """
#     lang = doc.vocab.lang
#     doc_items = []
#     for sent in doc.sents:
#         sent_items = [
#             Item(
#                 tok=tok,
#                 text=tok.text,
#                 ws=tok.whitespace_,
#                 pos=tok.pos_,
#                 is_word=not (tok.is_stop or tok.is_punct),
#             )
#             for tok in sent
#         ]
#         synonyms = {
#             (item.text, item.pos): rs.get_synonyms(item.text, lang=lang, sense=item.pos)
#             for item in sent_items
#             if item.is_word
#         }
#         # only keep items with non-empty synonym lists
#         synonyms = dicttoolz.valfilter(lambda v: v, synonyms)
#         sent_items = replace_with_synonyms(sent_items, synonyms, n_replacements)
#         sent_items = insert_synonyms(sent_items, synonyms, n_insertions)
#         sent_items = swap_items(sent_items, n_swaps)
#         sent_items = delete_items(sent_items, delete_prob)
#         doc_items.append(sent_items)
#     if shuffle_sents is True:
#         random.shuffle(doc_items)
#     augmented_text = "".join(
#         "".join(item.text + item.ws for item in sent_items)
#         for sent_items in doc_items
#     )
#     return make_spacy_doc(augmented_text, lang=lang)


def substitute_synonyms(aug_toks, num):
    """
    Randomly substitute tokens for which synonyms are available
    with a randomly selected synonym, up to ``num`` times or
    with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through synonym substitution.
        num (int or float): If int, maximum number of tokens with available synonyms
            to substitute with a randomly selected synonym; if float, probability
            that a given token with synonyms will be substituted.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idxs = [
        idx for idx, aug_tok in enumerate(aug_toks)
        if aug_tok.syns
    ]
    if not cand_idxs:
        return aug_toks[:]

    rand_idxs = set(_get_random_candidates(cand_idxs, num))
    if not rand_idxs:
        return aug_toks[:]

    new_aug_toks = []
    for idx, aug_tok in enumerate(aug_toks):
        if idx not in rand_idxs:
            new_aug_toks.append(aug_tok)
        else:
            new_aug_toks.append(
                AugTok(
                    text=random.choice(aug_tok.syns),
                    ws=aug_tok.ws,
                    pos=aug_tok.pos,
                    syns=aug_tok.syns,  # TODO: re-fetch syns? use []?
                    is_punct=aug_tok.is_punct,
                )
            )
    return new_aug_toks


def insert_synonyms(aug_toks, num):
    """
    Randomly insert random synonyms of tokens for which synonyms are available,
    up to ``num`` times or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through synonym insertion.
        num (int or float): If int, maximum number of tokens with available synonyms
            for which a random synonym is inserted; if float, probability
            that a given token with synonyms will provide a synonym for insertion.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    # bail out on very short sentences to avoid clobbering their meaning
    if len(aug_toks) < 3:
        return aug_toks[:]

    cand_aug_toks = [aug_tok for aug_tok in aug_toks if aug_tok.syns]
    if not cand_aug_toks:
        return aug_toks[:]

    rand_aug_toks = _get_random_candidates(cand_aug_toks, num)
    rand_idxs = random.sample(range(len(aug_toks)), len(rand_aug_toks))
    if not rand_idxs:
        return aug_toks[:]

    rand_idx_aug_toks = {
        rand_idx: rand_aug_tok
        for rand_idx, rand_aug_tok in zip(rand_idxs, rand_aug_toks)
    }
    new_aug_toks = []
    for idx, aug_tok in enumerate(aug_toks):
        if idx not in rand_idx_aug_toks:
            new_aug_toks.append(aug_tok)
        else:
            rand_aug_tok = rand_idx_aug_toks[idx]
            new_aug_toks.append(
                AugTok(
                    text=random.choice(rand_aug_tok.syns),
                    ws=" ",
                    pos=rand_aug_tok.pos,
                    syns=rand_aug_tok.syns, # TODO: re-fetch syns? use []?
                    is_punct=aug_tok.is_punct,
                )
            )
            new_aug_toks.append(aug_tok)
    return new_aug_toks


def swap_tokens(aug_toks, num):
    """
    Randomly swap the positions of two tokens with the same part-of-speech tag,
    up to ``num`` times or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through swapping tokens with the same part of speech.
        num (int or float): If int, maximum number of same-POS token pairs to swap;
            if float, probability that a given token pair will be swapped.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idx_pairs = list(
        itertools.chain.from_iterable(
            itertools.combinations(
                (
                    idx for idx, aug_tok in enumerate(aug_toks)
                    if aug_tok.pos == pos
                ),
                2,
            )
            for pos in ("NOUN", "VERB", "ADJ", "ADV")
        )
    )
    if not cand_idx_pairs:
        return aug_toks[:]

    rand_idx_pairs = _get_random_candidates(cand_idx_pairs, num)
    if not rand_idx_pairs:
        return aug_toks[:]

    new_aug_toks = aug_toks[:]
    for idx1, idx2 in rand_idx_pairs:
        at1 = new_aug_toks[idx1]
        at2 = new_aug_toks[idx2]
        new_aug_toks[idx1] = AugTok(
            text=at2.text,
            ws=at1.ws,
            pos=at2.pos,
            syns=at2.syns,
            is_punct=at2.is_punct,
        )
        new_aug_toks[idx2] = AugTok(
            text=at1.text,
            ws=at2.ws,
            pos=at1.pos,
            syns=at1.syns,
            is_punct=at1.is_punct,
        )
    return new_aug_toks


def delete_tokens(aug_toks, num):
    """
    Randomly delete non-punctuation tokens, up to ``num`` times
    or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through token deletion.
        num (int or float): If int, maximum number of non-punctuation tokens
            to delete; if float, probability that a given token will be deleted.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idxs = [
        idx for idx, aug_tok in enumerate(aug_toks)
        if not aug_tok.is_punct
    ]
    if not cand_idxs:
        return aug_toks[:]

    rand_idxs = set(_get_random_candidates(cand_idxs, num))
    if not rand_idxs:
        return aug_toks[:]

    return [
        aug_tok for idx, aug_tok in enumerate(aug_toks)
        if idx not in rand_idxs
    ]


def substitute_chars(aug_toks, num, char_weights):
    """
    Randomly substitute a character with another in randomly selected tokens,
    up to ``num`` times or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through character substitution.
        num (int or float): If int, maximum number of tokens to modify
            with a random character substitution; if float, probability
            that a given token will be modified.
        char_weights (List[Tuple[str, int]]): Collection of (character, weight) pairs,
            used to perform weighted random selection of characters to substitute
            into selected tokens. Characters with higher weight are
            more likely to be substituted.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idxs = [
        idx for idx, aug_tok in enumerate(aug_toks)
        if not (aug_tok.is_punct or len(aug_tok.text) < 3)
    ]
    if not cand_idxs:
        return aug_toks[:]

    rand_idxs = set(_get_random_candidates(cand_idxs, num))
    if not rand_idxs:
        return aug_toks[:]

    rand_chars = iter(
        random.choices(
            [char for char, _ in char_weights],
            weights=[weight for _, weight in char_weights],
            k=len(rand_idxs),
        )
    )
    new_aug_toks = []
    for idx, aug_tok in enumerate(aug_toks):
        if idx not in rand_idxs:
            new_aug_toks.append(aug_tok)
        else:
            text = list(aug_tok.text)
            rand_char_idx = random.choice(range(len(text)))
            rand_char = next(rand_chars)
            text[rand_char_idx] = rand_char
            new_aug_toks.append(
                AugTok(
                    text="".join(text),
                    ws=aug_tok.ws,
                    pos=aug_tok.pos,
                    syns=aug_tok.syns,
                    is_punct=aug_tok.is_punct,
                )
            )
    return new_aug_toks


def insert_chars(aug_toks, num, char_weights):
    """
    Randomly insert a character into randomly selected tokens,
    up to ``num`` times or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through character insertion.
        num (int or float): If int, maximum number of tokens to modify
            with a random character insertion; if float, probability
            that a given token will be modified.
        char_weights (List[Tuple[str, int]]): Collection of (character, weight) pairs,
            used to perform weighted random selection of characters to insert
            into selected tokens. Characters with higher weight are
            more likely to be inserted.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idxs = [
        idx for idx, aug_tok in enumerate(aug_toks)
        if not (aug_tok.is_punct or len(aug_tok.text) < 3)
    ]
    if not cand_idxs:
        return aug_toks[:]

    rand_idxs = set(_get_random_candidates(cand_idxs, num))
    if not rand_idxs:
        return aug_toks[:]

    rand_chars = iter(
        random.choices(
            [char for char, _ in char_weights],
            weights=[weight for _, weight in char_weights],
            k=len(rand_idxs),
        )
    )
    new_aug_toks = []
    for idx, aug_tok in enumerate(aug_toks):
        if idx not in rand_idxs:
            new_aug_toks.append(aug_tok)
        else:
            text = list(aug_tok.text)
            rand_char_idx = random.choice(range(len(text)))
            rand_char = next(rand_chars)
            text.insert(rand_char_idx, rand_char)
            new_aug_toks.append(
                AugTok(
                    text="".join(text),
                    ws=aug_tok.ws,
                    pos=aug_tok.pos,
                    syns=aug_tok.syns,
                    is_punct=aug_tok.is_punct,
                )
            )
    return new_aug_toks


def swap_chars(aug_toks, num):
    """
    Randomly swap two *adjacent* characters in randomly-selected, non-punctuation tokens,
    up to ``num`` times or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through character swapping.
        num (int or float): If int, maximum number of tokens to modify
            with a random character swap; if float, probability
            that a given token will be modified.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idxs = [
        idx for idx, aug_tok in enumerate(aug_toks)
        if not (aug_tok.is_punct or len(aug_tok.text) < 3)
    ]
    if not cand_idxs:
        return aug_toks[:]

    rand_idxs = set(_get_random_candidates(cand_idxs, num))
    if not rand_idxs:
        return aug_toks[:]

    new_aug_toks = []
    for idx, aug_tok in enumerate(aug_toks):
        if idx not in rand_idxs:
            new_aug_toks.append(aug_tok)
        else:
            text = list(aug_tok.text)
            idx = random.choice(range(1, len(text)))
            text[idx - 1], text[idx] = text[idx], text[idx - 1]
            new_aug_toks.append(
                AugTok(
                    text="".join(text),
                    ws=aug_tok.ws,
                    pos=aug_tok.pos,
                    syns=aug_tok.syns,
                    is_punct=aug_tok.is_punct,
                )
            )
    return new_aug_toks


def delete_chars(aug_toks, num):
    """
    Randomly delete a character in randomly-selected, non-punctuation tokens,
    up to ``num`` times or with a probability of ``num``.

    Args:
        aug_toks (List[:class:`AugTok`]): Sequence of tokens to augment
            through character deletion.
        num (int or float): If int, maximum number of tokens to modify
            with a random character deletion; if float, probability
            that a given token will be modified.

    Returns:
        List[:class:`AugTok`]: New, augmented sequence of tokens.
    """
    _validate_aug_toks(aug_toks)
    cand_idxs = [
        idx for idx, aug_tok in enumerate(aug_toks)
        if not (aug_tok.is_punct or len(aug_tok.text) < 3)
    ]
    if not cand_idxs:
        return aug_toks[:]

    rand_idxs = set(_get_random_candidates(cand_idxs, num))
    if not rand_idxs:
        return aug_toks[:]

    new_aug_toks = []
    for idx, aug_tok in enumerate(aug_toks):
        if idx not in rand_idxs:
            new_aug_toks.append(aug_tok)
        else:
            rand_cidx = random.choice(range(len(aug_tok.text)))
            new_aug_toks.append(
                AugTok(
                    text="".join(char for cidx, char in enumerate(aug_tok.text) if cidx != rand_cidx),
                    ws=aug_tok.ws,
                    pos=aug_tok.pos,
                    syns=aug_tok.syns,
                    is_punct=aug_tok.is_punct,
                )
            )
    return new_aug_toks


def to_aug_toks(spacy_obj):
    """
    Cast a sequence of spaCy ``Token`` s to a list of ``AugTok`` objects,
    suitable for use in data augmentation transform functions.

    Args:
        spacy_obj (:class:`spacy.tokens.Doc` or :class:`spacy.tokens.Span`)

    Returns:
        List[List[:class:`AugTok`]]
    """
    if not isinstance(spacy_obj, (Doc, Span)):
        raise TypeError(
            "`spacy_obj` must be of type {}, not {}".format((Doc, Span), type(spacy_obj))
        )
    if isinstance(spacy_obj, Doc) and spacy_obj.is_sentenced:
        return [_to_flat_aug_toks(sent) for sent in spacy_obj.sents]
    else:
        return [_to_flat_aug_toks(spacy_obj)]


def _to_flat_aug_toks(spacy_obj):
    """
    Args:
        spacy_obj (:class:`spacy.tokens.Doc` or :class:`spacy.tokens.Span`)

    Returns:
        List[:class:`AugTok`]
    """
    if not isinstance(spacy_obj, (Doc, Span)):
        raise TypeError(
            "`spacy_obj` must be of type {}, not {}".format((Doc, Span), type(spacy_obj)))
    lang = spacy_obj.vocab.lang
    return [
        AugTok(
            text=tok.text,
            ws=tok.whitespace_,
            pos=tok.pos_,
            syns=(
                rs.get_synonyms(tok.text, lang=lang, sense=tok.pos_)
                if not (tok.is_stop or tok.is_punct)
                else []
            ),
            is_punct=tok.is_punct,
        )
        for tok in spacy_obj
    ]


def _validate_aug_toks(aug_toks):
    if not (isinstance(aug_toks, list) and isinstance(aug_toks[0], AugTok)):
        raise TypeError(
            "aug_toks must be of type List[AugTok], not {}[{}]".format(
                type(aug_toks), type(aug_toks[0]))
        )


def _get_random_candidates(cands, num):
    """
    Args:
        cands (List[obj])
        num (int or float)

    Returns:
        List[obj]
    """
    if isinstance(num, int) and num >= 0:
        rand_cands = random.sample(cands, min(num, len(cands)))
    elif isinstance(num, float) and 0.0 <= num <= 1.0:
        rand_cands = [cand for cand in cands if random.random() < num]
    else:
        raise ValueError(
            "num={} is invalid; must be an int >= 0 or a float in [0.0, 1.0]".format(num)
        )
    return rand_cands
