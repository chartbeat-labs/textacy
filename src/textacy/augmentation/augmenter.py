from __future__ import annotations

import random
from typing import Optional, Sequence

from spacy.tokens import Doc

from .. import spacier, types, utils
from . import utils as aug_utils


class Augmenter:
    """
    Randomly apply one or many data augmentation transforms to spaCy ``Doc`` s
    to produce new docs with additional variety and/or noise in the data.

    Initialize an ``Augmenter`` with multiple transforms, and customize the randomization
    of their selection when applying to a document::

        >>> tfs = [transforms.delete_words, transforms.swap_chars, transforms.delete_chars]
        >>> Augmenter(tfs, num=None)  # all tfs applied each time
        >>> Augmenter(tfs, num=1)  # one randomly-selected tf applied each time
        >>> Augmenter(tfs, num=0.5)  # tfs randomly selected with 50% prob each time
        >>> augmenter = Augmenter(tfs, num=[0.4, 0.8, 0.6])  # tfs randomly selected with 40%, 80%, 60% probs, respectively, each time

    Apply transforms to a given ``Doc`` to produce new documents::

        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> doc = textacy.make_spacy_doc(text, lang="en_core_web_sm")
        >>> augmenter.apply_transforms(doc, lang="en_core_web_sm")
        The quick brown ox jupms over the lazy dog.
        >>> augmenter.apply_transforms(doc, lang="en_core_web_sm")
        The quikc brown fox over the lazy dog.
        >>> augmenter.apply_transforms(doc, lang="en_core_web_sm")
        quick brown fox jumps over teh lazy dog.

    Parameters for individual transforms may be specified when initializing ``Augmenter``
    or, if necessary, when applying to individual documents::

        >>> from functools import partial
        >>> tfs = [partial(transforms.delete_words, num=3), transforms.swap_chars]
        >>> augmenter = Augmenter(tfs)
        >>> augmenter.apply_transforms(doc, lang="en_core_web_sm")
        brown fox jumps over layz dog.
        >>> augmenter.apply_transforms(doc, lang="en_core_web_sm", pos={"NOUN", "ADJ"})
        The jumps over the lazy odg.

    Args:
        transforms: Ordered sequence of callables that must take list[:obj:`AugTok`]
            as their first positional argument and return another list[:obj:`AugTok`].

            .. note:: Although the particular transforms applied may vary doc-by-doc,
               they are applied *in order* as listed here. Since some transforms may
               clobber text in a way that makes other transforms less effective,
               a stable ordering can improve the quality of augmented data.

        num: If int, number of transforms to randomly select from ``transforms`` each time
            :meth:`Augmenter.apply_tranforms()` is called.
            If float, probability that any given transform will be selected.
            If Sequence[float], the probability that the corresponding transform
            in ``transforms`` will be selected (these must be the same length).
            If None (default), num is set to ``len(transforms)``, which means that
            every transform is applied each time.

    See Also:
        A collection of general-purpose transforms are implemented in
        :mod:`textacy.augmentation.transforms`.
    """

    def __init__(
        self,
        transforms: Sequence[types.AugTransform],
        *,
        num: Optional[int | float | Sequence[float]] = None,
    ):
        self.tfs = self._validate_transforms(transforms)
        self.num = self._validate_num(num)

    def apply_transforms(self, doc: Doc, lang: types.LangLike, **kwargs) -> Doc:
        """
        Sequentially apply some subset of data augmentation transforms to ``doc``,
        then return a new ``Doc`` created from the augmented text using ``lang``.

        Args:
            doc
            lang
            **kwargs: If, for whatever reason, you have to pass keyword argument values
                into transforms that vary or depend on characteristics of ``doc``,
                specify them here. The transforms' call signatures will be inspected,
                and values will be passed along, as needed.

        Returns:
            :class:`spacy.tokens.Doc`
        """
        if doc.has_annotation("SENT_START"):
            nested_aug_toks = [aug_utils.to_aug_toks(sent) for sent in doc.sents]
        else:
            nested_aug_toks = [aug_utils.to_aug_toks(doc)]
        tfs = self._get_random_transforms()
        new_nested_aug_toks = []
        for aug_toks in nested_aug_toks:
            # this is a bit of a hack, but whatchagonnado
            if kwargs:
                for tf in tfs:
                    tf_kwargs = utils.get_kwargs_for_func(tf, kwargs)
                    aug_toks = tf(aug_toks, **tf_kwargs)
            else:
                for tf in tfs:
                    aug_toks = tf(aug_toks)
            new_nested_aug_toks.append(aug_toks)
        return self._make_new_spacy_doc(new_nested_aug_toks, lang)

    def _validate_transforms(
        self, transforms: Sequence[types.AugTransform]
    ) -> tuple[types.AugTransform, ...]:
        transforms = tuple(transforms)
        if not transforms:
            raise ValueError("at least one transform callable must be specified")
        elif not all(callable(transform) for transform in transforms):
            raise TypeError("all transforms must be callable")
        else:
            return transforms

    def _validate_num(
        self, num: Optional[int | float | Sequence[float]]
    ) -> int | float | tuple[float, ...]:
        if num is None:
            return len(self.tfs)
        elif isinstance(num, int) and 0 <= num <= len(self.tfs):
            return num
        elif isinstance(num, float) and 0.0 <= num <= 1.0:
            return num
        elif (
            isinstance(num, (tuple, list))
            and len(num) == len(self.tfs)
            and all(isinstance(n, float) and 0.0 <= n <= 1.0 for n in num)
        ):
            return tuple(num)
        else:
            raise ValueError(
                f"num={num} is invalid; must be an int >= 1, a float in [0.0, 1.0], "
                "or a list of floats of length equal to given transforms"
            )

    def _get_random_transforms(self) -> list[types.AugTransform]:
        num = self.num
        if isinstance(num, int):
            rand_idxs = random.sample(range(len(self.tfs)), min(num, len(self.tfs)))
            rand_tfs = [self.tfs[idx] for idx in sorted(rand_idxs)]
        elif isinstance(num, float):
            rand_tfs = [tf for tf in self.tfs if random.random() < num]
        else:
            rand_tfs = [
                tf for tf, tf_num in zip(self.tfs, num) if random.random() < tf_num
            ]
        return rand_tfs

    def _make_new_spacy_doc(self, nested_aug_tokens, lang: types.LangLike) -> Doc:
        # TODO: maybe collect words, spaces, and array vals
        # then directly instantiate a new Doc object?
        # this would require adding an array field to AugTok
        new_text = "".join(
            aug_tok.text + aug_tok.ws
            for aug_toks in nested_aug_tokens
            for aug_tok in aug_toks
        )
        return spacier.core.make_spacy_doc(new_text, lang=lang)
