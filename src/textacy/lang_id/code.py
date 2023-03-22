from typing import Optional

from spacy.tokens import Doc
from spacy.util import DummyTokenizer, registry
from spacy.vocab import Vocab


class CharTokenizer(DummyTokenizer):
    def __init__(
        self, vocab: Vocab, max_chars: Optional[int] = None, lower_case: bool = False
    ):
        self.vocab = vocab
        self.max_chars = max_chars
        self.lower_case = lower_case

    def __call__(self, text):
        if self.max_chars is not None:
            text = text[: self.max_chars]
        if self.lower_case is True:
            text = text.lower()
        words = list(text)
        spaces = [False] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


@registry.tokenizers("textacy.char_tokenizer.v1")
def create_char_tokenizer(max_chars: Optional[int], lower_case: bool):
    def create_tokenizer(nlp):
        return CharTokenizer(nlp.vocab, max_chars=max_chars, lower_case=lower_case)

    return create_tokenizer
