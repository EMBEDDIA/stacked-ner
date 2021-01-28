# -*- coding: utf-8 -*-

from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder
from torch import nn
import torch
import torch.nn.functional as F


class StackedTransformersCRF(nn.Module):
    def __init__(self, tag_vocabs, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans', bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):

        super().__init__()

        self.embed = embed
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        self.tag_vocabs = []
        self.out_fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()

        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            out_fc = nn.Linear(1536, len(tag_vocabs[i]))
            self.out_fcs.append(out_fc)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type='bioes', include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)

        self.fc_dropout = nn.Dropout(fc_dropout)

    def _forward(self, words, target=None, bigrams=None, seq_len=None):

        torch.cuda.empty_cache()

        mask = words.ne(0)
        words = self.embed(words)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            words = torch.cat([words, bigrams], dim=-1)

        targets = [target]

        chars = self.in_fc(words)
        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)

        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(self.out_fcs[i](chars), dim=-1))

        torch.cuda.empty_cache()

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[
                        0]
                else:
                    results['pred' + str(i)] = torch.argmax(logits[i], 2)
            return results

    def forward(self, words, target=None, seq_len=None):
        return self._forward(words, target, seq_len)

    def predict(self, words, seq_len=None):
        return self._forward(words, target=None)


class BertCRF(nn.Module):
    def __init__(self, embed, tag_vocabs, encoding_type='bio'):
        super().__init__()
        self.embed = embed
        self.tag_vocabs = []
        self.fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()

        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            linear = nn.Linear(self.embed.embed_size, len(tag_vocabs[i]))
            self.fcs.append(linear)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type=encoding_type, include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)

    def _forward(self, words, target=None, seq_len=None):
        mask = words.ne(0)
        words = self.embed(words)

        targets = [target]

        words_fcs = []
        for i in range(len(targets)):
            words_fcs.append(self.fcs[i](words))

        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(words_fcs[i], dim=-1))

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[
                        0]
                else:
                    results['pred' + str(i)] = torch.argmax(logits[i], 2)

            return results

    def forward(self, words, target=None, seq_len=None):
        return self._forward(words, target, seq_len)

    def predict(self, words, seq_len=None):
        return self._forward(words, target=None)
