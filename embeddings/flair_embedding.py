# -*- coding: utf-8 -*-


__all__ = [
    "FlairEmbedding",
]

import torch

from fastNLP.embeddings.contextual_embedding import ContextualEmbedding
from fastNLP.core import Vocabulary
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence


class FlairEmbedding(ContextualEmbedding):

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False, **kwargs):

        super(FlairEmbedding, self).__init__(
            vocab, word_dropout=word_dropout, dropout=dropout)

        if word_dropout > 0:
            assert vocab.unknown is not None, "When word_drop>0, Vocabulary must contain the unknown token."

        self._word_sep_index = -100
        if '[SEP]' in vocab:
            self._word_sep_index = vocab['[SEP]']
        self._word_cls_index = -100
        if '[CLS]' in vocab:
            self._word_cls_index = vocab['CLS']

        self.vocab = vocab

        self.model = FlairEmbeddings(model=model_dir_or_name,
                                     fine_tune=False)

        self.requires_grad = requires_grad
        self._embed_size = self.model.embedding_length

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):

        max_length = words.shape[1]
        words = self.drop_word(words)
        words_sentences = []
        for sentence in words:
            words_sentences.append([self.vocab.idx2word[word.item()]
                                    for word in sentence if word.item() != 0])

        words = [Sentence(' '.join(x)) for x in words_sentences]
        self.model.embed(words)

        outputs = torch.stack([torch.stack([x.embedding for x in y] + (max_length - len(
            y)) * [torch.zeros(2048).to(next(self.parameters()).device)]) for y in words])

        del words
        del words_sentences
        torch.cuda.empty_cache()

        if outputs is not None:
            return self.dropout(outputs)
        return self.dropout(outputs)

    def drop_word(self, words):

        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                mask = torch.full_like(
                    words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)
                pad_mask = words.ne(self._word_pad_index)
                mask = pad_mask.__and__(mask)
                if self._word_sep_index != -100:
                    not_sep_mask = words.ne(self._word_sep_index)
                    mask = mask.__and__(not_sep_mask)
                if self._word_cls_index != -100:
                    not_cls_mask = words.ne(self._word_cls_index)
                    mask = mask.__and__(not_cls_mask)
                words = words.masked_fill(mask, self._word_unk_index)
        return words
