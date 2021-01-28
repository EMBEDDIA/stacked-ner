# -*- coding: utf-8 -*-
"""https://github.com/fastnlp/fastNLP"""
__all__ = [
    "BertEmbedding",
    "BertWordPieceEncoder"
]

import warnings
from itertools import chain
from functools import partial

import numpy as np
import torch
from torch import nn

from fastNLP.embeddings.contextual_embedding import ContextualEmbedding
from fastNLP.core import logger, Vocabulary
from fastNLP.io.file_utils import PRETRAINED_BERT_MODEL_DIR
from fastNLP.modules.encoder.bert import BertModel
from fastNLP.modules.tokenizer import BertTokenizer
# frot transformers import BertTokenizer
import collections


class BertEmbedding(ContextualEmbedding):

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False, **kwargs):

        super(BertEmbedding, self).__init__(
            vocab, word_dropout=word_dropout, dropout=dropout)

        if word_dropout > 0:
            assert vocab.unknown is not None, "When word_drop>0, Vocabulary must contain the unknown token."

        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in ('first', 'last'):
                logger.warning("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                               " faster speed.")
                warnings.warn("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                              " faster speed.")

        self._word_sep_index = -100
        if '[SEP]' in vocab:
            self._word_sep_index = vocab['[SEP]']
        self._word_cls_index = -100
        if '[CLS]' in vocab:
            self._word_cls_index = vocab['CLS']

        only_use_pretrain_bpe = kwargs.get('only_use_pretrain_bpe', False)
        truncate_embed = kwargs.get('truncate_embed', True)
        min_freq = kwargs.get('min_freq', 2)

        self.model = _BertWordModel(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                    pool_method=pool_method, include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls, auto_truncate=auto_truncate, min_freq=min_freq,
                                    only_use_pretrain_bpe=only_use_pretrain_bpe, truncate_embed=truncate_embed)

        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * \
            self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):

        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)
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


class BertWordPieceEncoder(nn.Module):

    def __init__(self, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1', pooled_cls: bool = False,
                 word_dropout=0, dropout=0, requires_grad: bool = True):

        super().__init__()

        self.model = _BertWordPieceModel(
            model_dir_or_name=model_dir_or_name, layers=layers, pooled_cls=pooled_cls)
        self._sep_index = self.model._sep_index
        self._cls_index = self.model._cls_index
        self._wordpiece_pad_index = self.model._wordpiece_pad_index
        self._wordpiece_unk_index = self.model._wordpiece_unknown_index
        self._embed_size = len(self.model.layers) * \
            self.model.encoder.hidden_size
        self.requires_grad = requires_grad
        self.word_dropout = word_dropout
        self.dropout_layer = nn.Dropout(dropout)

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def embedding_dim(self):
        return self._embed_size

    @property
    def num_embedding(self):
        return self.model.encoder.config.vocab_size

    def index_datasets(self, *datasets, field_name, add_cls_sep=True):

        self.model.index_datasets(
            *datasets, field_name=field_name, add_cls_sep=add_cls_sep)

    def forward(self, word_pieces, token_type_ids=None):

        if token_type_ids is None:
            with torch.no_grad():
                sep_mask = word_pieces.eq(
                    self._sep_index)
                sep_mask_cumsum = sep_mask.long().flip(
                    dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():
                    token_type_ids = token_type_ids.eq(0).long()

        word_pieces = self.drop_word(word_pieces)
        outputs = self.model(word_pieces, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout_layer(outputs)

    def drop_word(self, words):

        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                not_sep_mask = words.ne(self._sep_index)
                not_cls_mask = words.ne(self._cls_index)
                replaceable_mask = not_sep_mask.__and__(not_cls_mask)
                mask = torch.full_like(
                    words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)
                pad_mask = words.ne(self._wordpiece_pad_index)
                mask = pad_mask.__and__(mask).__and__(
                    replaceable_mask)
                words = words.masked_fill(mask, self._wordpiece_unk_index)
        return words


additional_special_tokens = ['<e1>', '</e1>']


class _BertWordModel(nn.Module):
    def __init__(self, model_dir_or_name: str, vocab: Vocabulary, layers: str = '-1', pool_method: str = 'first',
                 include_cls_sep: bool = False, pooled_cls: bool = False, auto_truncate: bool = False, min_freq=2,
                 only_use_pretrain_bpe=False, truncate_embed=True):
        super().__init__()

        # , additional_special_tokens=additional_special_tokens)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = BertModel.from_pretrained(model_dir_or_name)

        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        encoder_layer_number = len(self.encoder.encoder.layer)
        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a bert model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        logger.info("Vocab is not used anymore.")
        logger.info("Start to generate word pieces for word.")
        self._has_sep_in_vocab = '[SEP]' in vocab
#
#        word_piece_dict = {'[CLS]': 1, '[SEP]': 1}
#        new_add_to_bpe_vocab = 0
#
#        unsegment_count = 0
#        if '[sep]' in vocab:
#            warnings.warn(
#                "Lower cased [sep] detected, it cannot be correctly recognized as [SEP] by BertEmbedding.")
#        if "[CLS]" in vocab:
#            warnings.warn("[CLS] detected in your vocabulary. BertEmbedding will add [CLS] and [SEP] to the begin "
#                          "and end of the input automatically, make sure you don't add [CLS] and [SEP] at the begin"
#                          " and end.")
#
#        print('Len vocab:', len(vocab))
#        for word, index in vocab:
#            if index == vocab.padding_idx:
#                word = '[PAD]'
#            elif index == vocab.unknown_idx:
#                word = '[UNK]'
#            _words = self.tokenizer.basic_tokenizer._tokenize_chinese_chars(
#                word).split()
#            word_pieces = []
#            for w in _words:
#                word_pieces.extend(
#                    self.tokenizer.wordpiece_tokenizer.tokenize(w))
#            if len(word_pieces) == 1:
#                if not vocab._is_word_no_create_entry(word):
#                    if index != vocab.unknown_idx and word_pieces[0] == '[UNK]':
#                        if vocab.word_count[word] >= min_freq and not vocab._is_word_no_create_entry(
#                                word) and not only_use_pretrain_bpe:
#                            word_piece_dict[word] = 1
#                            new_add_to_bpe_vocab += 1
#                        unsegment_count += 1
#
#                        continue
#            for word_piece in word_pieces:
#                word_piece_dict[word_piece] = 1
#        original_embed = self.encoder.embeddings.word_embeddings.weight.data
#
#        if not truncate_embed:
#            word_piece_dict.update(self.tokenizer.vocab)
#
#        embed = nn.Embedding(len(word_piece_dict), original_embed.size(1))
#        new_word_piece_vocab = collections.OrderedDict()
#
#        for index, token in enumerate(['[PAD]', '[UNK]']):
#            index = word_piece_dict.pop(token, None)
#            if index is not None:
#                new_word_piece_vocab[token] = len(new_word_piece_vocab)
#                embed.weight.data[new_word_piece_vocab[token]
#                                  ] = original_embed[self.tokenizer.vocab[token]]
#        for token in word_piece_dict.keys():
#            if token not in new_word_piece_vocab:
#                new_word_piece_vocab[token] = len(new_word_piece_vocab)
#            index = new_word_piece_vocab[token]
#            if token in self.tokenizer.vocab:
#                embed.weight.data[index] = original_embed[self.tokenizer.vocab[token]]
#            else:
#                embed.weight.data[index] = original_embed[self.tokenizer.vocab['[UNK]']]
#
#        self.tokenizer._reinit_on_new_vocab(new_word_piece_vocab)
#        self.encoder.embeddings.word_embeddings = embed
#
#        self.encoder.config.vocab_size = len(new_word_piece_vocab)
#        if unsegment_count > 0:
#            if only_use_pretrain_bpe or new_add_to_bpe_vocab == 0:
#                logger.info(f"{unsegment_count} words are unsegmented.")
#            else:
#                logger.info(
# f"{unsegment_count} words are unsegmented. Among them,
# {new_add_to_bpe_vocab} added to the BPE vocab.")

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            word_pieces = self.tokenizer.wordpiece_tokenizer.tokenize(word)

            word_pieces = self.tokenizer.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))

        self._cls_index = self.tokenizer.vocab['[CLS]']
        self._sep_index = self.tokenizer.vocab['[SEP]']
        self._word_pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenizer.vocab['[PAD]']
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.register_buffer('word_pieces_lengths',
                             torch.LongTensor(word_pieces_lengths))
        logger.debug("Successfully generate word pieces.")

#        self.tokenizer.save_pretrained(model_dir_or_name)

    def forward(self, words):

        with torch.no_grad():
            batch_size, max_word_len = words.size()
            word_mask = words.ne(self._word_pad_index)
            seq_len = word_mask.sum(dim=-1)
            batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False),
                                                                                   0)
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)
            max_word_piece_length = batch_word_pieces_length.sum(
                dim=-1).max().item()
            if max_word_piece_length + 2 > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths + 2 > self._max_position_embeddings,
                        self._max_position_embeddings - 2)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            word_pieces = words.new_full((batch_size, min(max_word_piece_length + 2, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            attn_masks = torch.zeros_like(word_pieces)
            word_indexes = words.cpu().numpy()
            for i in range(batch_size):
                word_pieces_i = list(
                    chain(*self.word_to_wordpieces[word_indexes[i, :seq_len[i]]]))
                if self.auto_truncate and len(
                        word_pieces_i) > self._max_position_embeddings - 2:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings - 2]
                word_pieces[i, 1:word_pieces_lengths[i] +
                            1] = torch.LongTensor(word_pieces_i)
                attn_masks[i, :word_pieces_lengths[i] + 2].fill_(1)

            word_pieces[:, 0].fill_(self._cls_index)
            batch_indexes = torch.arange(batch_size).to(words)
            word_pieces[batch_indexes,
                        word_pieces_lengths + 1] = self._sep_index
            if self._has_sep_in_vocab:
                sep_mask = word_pieces.eq(self._sep_index).long()
                sep_mask_cumsum = sep_mask.flip(
                    dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():
                    token_type_ids = token_type_ids.eq(0).long()
            else:
                token_type_ids = torch.zeros_like(word_pieces)
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                 bert_outputs[-1].size(-1))

        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))
        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(
            batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:,
                                     1:] = batch_word_pieces_length.cumsum(dim=-1)

        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max(
            )]
            batch_word_pieces_cum_length.masked_fill_(
                batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand(
                (batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, 1:seq_len.max(
            ) + 1] - 1
            batch_word_pieces_cum_length.masked_fill_(
                batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand(
                (batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if max_word_piece_length > real_word_piece_length:
                paddings = output_layer.new_zeros(batch_size,
                                                  max_word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat(
                    (output_layer, paddings), dim=1).contiguous()
            truncate_output_layer = output_layer[:, 1:-1]
            if self.pool_method == 'first':
                tmp = truncate_output_layer[_batch_indexes,
                                            batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(
                    word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(
                    1) + s_shift] = tmp

            elif self.pool_method == 'last':
                tmp = truncate_output_layer[_batch_indexes,
                                            batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(
                    word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(
                    1) + s_shift] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i,
                                                                  j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift], _ = torch.max(
                            truncate_output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i,
                                                                  j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift] = torch.mean(
                            truncate_output_layer[i, start:end], dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len +
                        s_shift] = output_layer[batch_indexes, word_pieces_lengths + s_shift]

        return outputs


class _BertWordPieceModel(nn.Module):

    def __init__(self, model_dir_or_name: str, layers: str = '-1',
                 pooled_cls: bool = False):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = BertModel.from_pretrained(model_dir_or_name)
        encoder_layer_number = len(self.encoder.encoder.layer)

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")

        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                    f"a bert model with {encoder_layer_number} layers."

        self._cls_index = self.tokenizer.cls_index
        self._sep_index = self.tokenizer.sep_index
        self._wordpiece_unknown_index = self.tokenizer.unk_index
        self._wordpiece_pad_index = self.tokenizer.pad_index
        self.pooled_cls = pooled_cls

    def index_datasets(self, *datasets, field_name, add_cls_sep=True):

        encode_func = partial(self.tokenizer.encode,
                              add_special_tokens=add_cls_sep)

        for index, dataset in enumerate(datasets):
            try:
                dataset.apply_field(encode_func, field_name=field_name, new_field_name='word_pieces',
                                    is_input=True)
                dataset.set_pad_val('word_pieces', self._wordpiece_pad_index)
            except Exception as e:
                logger.error(
                    f"Exception happens when processing the {index} dataset.")
                raise e

    def forward(self, word_pieces, token_type_ids=None):

        batch_size, max_len = word_pieces.size()

        attn_masks = word_pieces.ne(self._wordpiece_pad_index)
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        outputs = bert_outputs[0].new_zeros(
            (len(self.layers), batch_size, max_len, bert_outputs[0].size(-1)))
        for l_index, l in enumerate(self.layers):
            bert_output = bert_outputs[l]
            if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                bert_output[:, 0] = pooled_cls
            outputs[l_index] = bert_output
        return outputs
