from typing import Iterable
from torchtext.vocab import build_vocab_from_iterator, Vocab


class Tokenizer:
    def __init__(self, special_tokens: list[str] = [], min_freq: int = 1) -> None:
        if special_tokens:
            special_tokens.extend(["<oov>"])
        else:
            special_tokens = ["<oov>"]

        self.special_tokens: list[str] = special_tokens
        self.min_freq: int = min_freq
        self.vocab: Vocab = None
        self._is_fit: bool = False

    def fit(self, corpus: Iterable[str | int]):
        vocab: Vocab = build_vocab_from_iterator(
            corpus,
            specials=self.special_tokens,
            min_freq=self.min_freq
        )
        default_index = vocab.get_stoi()["<oov>"]
        vocab.set_default_index(default_index)
        self.vocab = vocab
        self._is_fit = True
        return self

    def fit_transform(self, corpus):
        if not self._is_fit:
            self.fit(corpus)
        return self.transform(corpus)

    def transform(self, corpus):
        vocab = self.vocab
        return [
            vocab.lookup_indices(seq) for seq in corpus
        ]

    def inverse_transform(self, tokenized_corpus):
        vocab = self.vocab
        return [
            vocab.lookup_tokens(seq) for seq in tokenized_corpus
        ]


def get_tokenizer(
        text_data: Iterable[str | int],
        special_tokens: list[str] = [],
        min_freq: int = 1):

    return Tokenizer(special_tokens, min_freq).fit(text_data)
