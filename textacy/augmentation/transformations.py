import collections
import random

from cytoolz import dicttoolz

from .. import resources
from ..doc import make_spacy_doc


rs = resources.ConceptNet()
swap_pos = ["NOUN", "ADJ", "VERB"]

Item = collections.namedtuple("Item", ["tok", "text", "ws", "pos", "is_word"])


def apply(
    doc,
    n_replacements=1,
    n_insertions=1,
    n_swaps=1,
    delete_prob=0.05,
    shuffle_sents=True,
):
    """
    Apply a variety of transformations to the sentences in ``doc`` to generate
    a similar-but-different document, suitable for improving performance
    on text classification tasks.

    Args:
        doc (:class:`spacy.tokens.Doc`): Text document to be augmented through
            a variety of transformations.
        n_replacements (int): Maximum number of items to replace with synonyms,
            per sentence.
        n_insertions (int): Maximum number of times to insert synonyms, per sentence.
        n_swaps (int): Maximum number of times to swap items, per sentence.
        delete_prob (float): Probability that any given item is deleted.
        shuffle_sents (bool): If True, shuffle the order of sentences in ``doc``;
            otherwise, leave sentence order unchanged.

    Returns:
        :class:`spacy.tokens.Doc`: New, transformed document generated from ``doc``.

    References:
        Wei, Jason W., and Kai Zou. "Eda: Easy data augmentation techniques
        for boosting performance on text classification tasks."
        arXiv preprint arXiv:1901.11196 (2019).
    """
    lang = doc.vocab.lang
    doc_items = []
    for sent in doc.sents:
        sent_items = [
            Item(
                tok=tok,
                text=tok.text,
                ws=tok.whitespace_,
                pos=tok.pos_,
                is_word=not (tok.is_stop or tok.is_punct),
            )
            for tok in sent
        ]
        synonyms = {
            (item.text, item.pos): rs.get_synonyms(item.text, lang=lang, sense=item.pos)
            for item in sent_items
            if item.is_word
        }
        # only keep items with non-empty synonym lists
        synonyms = dicttoolz.valfilter(lambda v: v, synonyms)
        sent_items = replace_with_synonyms(sent_items, synonyms, n_replacements)
        sent_items = insert_synonyms(sent_items, synonyms, n_insertions)
        sent_items = swap_items(sent_items, n_swaps)
        sent_items = delete_items(sent_items, delete_prob)
        doc_items.append(sent_items)
    if shuffle_sents is True:
        random.shuffle(doc_items)
    augmented_text = "".join(
        "".join(item.text + item.ws for item in sent_items)
        for sent_items in doc_items
    )
    return make_spacy_doc(augmented_text, lang=lang)


def replace_with_synonyms(items, synonyms, n):
    """
    Randomly choose ``n`` items in ``items`` that aren't stop words and have synonyms,
    then replace each with a randomly selected synonym.

    Args:
        items (List[Item]): Sequence of items to augment through synonym replacement.
        synonyms (Dict[Tuple[str, str], List[str]]): Mapping of item (text, POS) to
            available synonyms' texts. Not all items in ``items`` have synonyms.
        n (int): Maximum number of items to replace with synonyms.

    Returns:
        List[Item]: Input ``items``, modified *in-place*.
    """
    candidate_idx_syns = [
        (idx, synonyms[(item.text, item.pos)])
        for idx, item in enumerate(items)
        if item.is_word and (item.text, item.pos) in synonyms
    ]
    if not candidate_idx_syns:
        return items

    random_idx_syns = random.sample(candidate_idx_syns, min(n, len(candidate_idx_syns)))
    for idx, syns in random_idx_syns:
        curr_item = items[idx]
        items[idx] = Item(
            tok=None,
            text=random.choice(syns),
            ws=curr_item.ws,
            pos=curr_item.pos,
            is_word=True,
        )
    return items


def insert_synonyms(items, synonyms, n):
    """
    Randomly select a synonym of a random item in ``items`` that's not a stop word,
    then randomly insert that synonym into ``items``, repeating the procedure ``n`` times.

    Args:
        items (List[Item]): Sequence of items to augment through synonym insertion.
        synonyms (Dict[Tuple[str, str], List[str]]): Mapping of item (text, POS) to
            available synonyms' texts. Not all items in ``items`` have synonyms.
        n (int): Maximum number of times to insert synonyms.

    Returns:
        List[Item]: Input ``items``, modified *in-place*.
    """
    random_synonyms = random.sample(synonyms.items(), min(n, len(synonyms)))
    if not random_synonyms or len(items) < 3:
        return items

    for (_, pos), syns in random_synonyms:
        random_syn = random.choice(syns)
        random_idx = random.randint(0, len(items) - 1)
        ws = items[random_idx - 1].ws
        # insert synonym into the list
        items.insert(
            random_idx,
            Item(tok=None, text=random_syn, ws=ws, pos=pos, is_word=True),
        )
        # we almost always want whitespace between this and the previous item
        # so fix it if we have to
        if random_idx > 0:
            prev_item = items[random_idx - 1]
            if not prev_item.ws:
                items[random_idx - 1] = Item(
                    tok=prev_item.tok,
                    text=prev_item.text,
                    ws=" ",
                    pos=prev_item.pos,
                    is_word=True,
                )
    return items


def swap_items(items, n):
    """
    Randomly swap the positions of two items in ``items`` with the same part-of-speech tag,
    repeating the procedure ``n`` times.

    Args:
        items (List[Item]): Sequence of items to augment through item swapping.
        n (int): Maximum number of times to swap items.

    Returns:
        List[Item]: Input ``items``, modified *in-place*.
    """
    for _ in range(n):
        random.shuffle(swap_pos)
        # only swap items of the same part-of-speech, to minimize ungrammatical results
        for pos in swap_pos:
            idxs = [idx for idx, item in enumerate(items) if item.pos == pos]
            if len(idxs) >= 2:
                idx1, idx2 = random.sample(idxs, 2)
                # can't do this, bc whitespace should not be swapped :/
                # items[idx2], items[idx1] = items[idx1], items[idx2]
                item1 = items[idx1]
                item2 = items[idx2]
                items[idx1] = Item(
                    tok=item2.tok,
                    text=item2.text,
                    ws=item1.ws,
                    pos=item2.pos,
                    is_word=item2.is_word,
                )
                items[idx2] = Item(
                    tok=item1.tok,
                    text=item1.text,
                    ws=item2.ws,
                    pos=item1.pos,
                    is_word=item1.is_word,
                )
                break
    return items


def delete_items(items, prob):
    """
    Randomly remove each item in ``items`` with probability ``prob``.

    Args:
        items (List[Item]): Sequence of items to augment through item deletion.
        prob (float): Probability that any given item is deleted.

    Returns:
        List[Item]: Input ``items``, modified *in-place*.
    """
    delete_idxs = [
        idx for idx, item in enumerate(items) if random.random() < prob and item.is_word
    ]
    for idx in sorted(delete_idxs, reverse=True):
        # TODO: determine if we want to adjust whitespace here
        # if idx > 0:
        #     prev_item = items[idx - 1]
        #     curr_item = items[idx]
        #     if prev_item.ws != curr_item.ws:
        #         items[idx - 1] = Item(
        #             tok=prev_item.tok,
        #             text=prev_item.text,
        #             ws=curr_item.ws,
        #             pos=prev_item.pos,
        #             is_word=prev_item.is_word,
        #         )
        del items[idx]
    return items
