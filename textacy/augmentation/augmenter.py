import inspect
import random

from spacy.tokens import Doc, Span

from ..doc import make_spacy_doc
from . import utils as aug_utils


class Augmenter:
    """
    TODO: DOCS

    Args:
        transforms (Sequence[Callable]): Ordered sequence of callables that must take
            a List[:obj:`AugTok`] as their first positional argument and return
            a List[:obj:`AugTok`].
        num (int or float or List[float]): If int, number of transforms to randomly select
            from ``transforms`` each time :meth:`Augmenter.apply_tranforms()` is called.
            If float, probability that any given transform will be selected.
            If List[float], the probability that the corresponding transform
            in ``transforms`` will be selected (these must be the same length).
            If None (default), num is set to ``len(transforms)``, which means that
            every transform is applied each time.

    See Also:
        A collection of good, general-purpose transforms are implemented in
        :mod:`textacy.augmentation.transforms`.
    """

    def __init__(self, transforms, *, num=None):
        self._tfs = self._validate_transforms(transforms)
        self._num = self._validate_num(num)

    def apply_transforms(self, doc, **kwargs):
        """
        Sequentially apply some subset of data augmentation transforms to ``doc``,
        then return a new ``Doc`` created from the augmented text.

        Args:
            doc (:class:`spacy.tokens.Doc`)
            **kwargs: If, for whatever reason, you have to pass keyword argument values
                into transforms that vary or depend on characteristics of ``doc``,
                specify them here. The transforms' call signatures will be inspected,
                and values will be passed along, as needed.

        Returns:
            :class:`spacy.tokens.Doc`
        """
        if doc.is_sentenced:
            nested_aug_toks = [aug_utils.to_aug_toks(sent) for sent in doc.sents]
        else:
            nested_aug_toks = [aug_utils.to_aug_toks(doc)]
        lang = doc.vocab.lang
        tfs = self._get_random_transforms()
        new_nested_aug_toks = []
        for aug_toks in nested_aug_toks:
            # this is a bit of a hack, but whatchagonnado
            if kwargs:
                for tf in tfs:
                    tf_params = inspect.signature(tf).parameters
                    tf_kwargs = {
                        kwarg: value for kwarg, value in kwargs.items()
                        if kwarg in tf_params
                    }
                    aug_toks = tf(aug_toks, **tf_kwargs)
            else:
                for tf in tfs:
                    aug_toks = tf(aug_toks)
            new_nested_aug_toks.append(aug_toks)
        return self._make_new_spacy_doc(new_nested_aug_toks, lang)

    def _validate_transforms(self, transforms):
        transforms = tuple(transforms)
        if not transforms:
            raise ValueError("at least one transform callable must be specified")
        elif not all(callable(transform) for transform in transforms):
            raise TypeError("all transforms must be callable")
        else:
            return transforms

    def _validate_num(self, num):
        if num is None:
            return len(self._tfs)
        elif isinstance(num, int) and 0 <= num <= len(self._tfs):
            return num
        elif isinstance(num, float) and 0.0 <= num <= 1.0:
            return num
        elif (
            isinstance(num, (tuple, list)) and
            len(num) == len(self._tfs) and
            all(isinstance(n, float) and 0.0 <= n <= 1.0 for n in num)
        ):
            return tuple(num)
        else:
            raise ValueError(
                "num={} is invalid; must be an int >= 1, a float in [0.0, 1.0], "
                "or a list of floats of length equal to given transforms".format(num)
            )

    def _get_random_transforms(self):
        num = self._num
        if isinstance(num, int):
            rand_idxs = random.sample(range(len(self._tfs)), min(num, len(self._tfs)))
            rand_tfs = [self._tfs[idx] for idx in sorted(rand_idxs)]
        elif isinstance(num, float):
            rand_tfs = [tf for tf in self._tfs if random.random() < num]
        else:
            rand_tfs = [
                tf for tf, tf_num in zip(self._tfs, self._num)
                if random.random() < tf_num
            ]
        return rand_tfs

    def _make_new_spacy_doc(self, nested_aug_tokens, lang):
        # TODO: maybe collect words, spaces, and array vals
        # then directly instantiate a new Doc object?
        # this would require adding an array field to AugTok
        new_text = "".join(
            aug_tok.text + aug_tok.ws
            for aug_toks in nested_aug_tokens
            for aug_tok in aug_toks
        )
        return make_spacy_doc(new_text, lang=lang)
