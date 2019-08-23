import collections
import random

from cytoolz import dicttoolz

from .. import resources
from ..doc import make_spacy_doc


Item = collections.namedtuple("Item", ["tok", "text", "ws", "pos", "is_word"])

rs = resources.ConceptNet()
swap_pos = ["NOUN", "ADJ", "VERB"]


def augment(
    doc,
    n_replacements=1,
    n_insertions=1,
    n_swaps=1,
    delete_prob=0.05,
    shuffle_sents=True,
):
    """
    Args:
        doc (:class:`spacy.tokens.Doc`)
        n_replacements (int)
        n_insertions (int)
        n_swaps (int)
        delete_prob (float)
        shuffle_sents (bool)

    Returns:
        :class:`spacy.tokens.Doc`
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
    Args:
        items (List[Item])
        synonyms (Dict[str, List[str]])
        n (int)

    Returns:
        List[Item]
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
    Args:
        items (List[Item])
        synonyms (Dict[str, List[str]])
        n (int)

    Returns:
        List[Item]
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
    Args:
        items (List[Item])
        n (int)

    Returns:
        List[Item]
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
    Args:
        items (List[Item])
        prob (float)

    Returns:
        List[Item]
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
