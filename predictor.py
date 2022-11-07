# -*- coding: utf-8 -*-
"""Partially https://github.com/fastnlp/fastNLP"""
__all__ = [
    "Predictor"
]

from collections import defaultdict

import torch
from fastNLP import DataSet
from fastNLP import DataSetIter
from fastNLP import SequentialSampler
from fastNLP.core.utils import _build_args, _move_dict_value_to_device, _get_model_device

from tqdm import tqdm


class Predictor(object):

    def __init__(self, network):

        if not isinstance(network, torch.nn.Module):
            raise ValueError(
                "Only fastNLP.models.BaseModel or torch.nn.Module is allowed, not {}".format(type(network)))
        self.network = network
        self.batch_size = 1
        self.batch_output = []

    def predict(self, data: DataSet, seq_len_field_name=None):
        r"""
        """
        if not isinstance(data, DataSet):
            raise ValueError(
                "Only Dataset class is allowed, not {}.".format(type(data)))
        if seq_len_field_name is not None and seq_len_field_name not in data.field_arrays:
            raise ValueError("Field name {} not found in DataSet {}.".format(
                seq_len_field_name, data))

        self.network.eval()  # self.network.module for multi-GPU
        network_device = _get_model_device(self.network)
        batch_output = defaultdict(list)
        data_iterator = DataSetIter(
            data, batch_size=self.batch_size, sampler=SequentialSampler(), as_numpy=False)

        # predict_func = self.network.module.predict  # self.network.module for
        # multi-GPU
        try:
            predict_func = self.network.predict
        except Exception:
            predict_func = self.network.module.predict

        with torch.no_grad():
            #            for batch_x, _ in tqdm(data_iterator):
            for batch_x, _ in tqdm(data_iterator, total=len(data_iterator)):
                _move_dict_value_to_device(batch_x, _, device=network_device)
                refined_batch_x = _build_args(predict_func, **batch_x)
                prediction = predict_func(**refined_batch_x)
                if seq_len_field_name is not None:
                    seq_lens = batch_x[seq_len_field_name].tolist()

                for key, value in prediction.items():
                    value = value.cpu().numpy()
                    if len(value.shape) == 1 or (
                            len(value.shape) == 2 and value.shape[1] == 1):
                        batch_output[key].extend(value.tolist())
                    else:
                        if seq_len_field_name is not None:
                            tmp_batch = []
                            for idx, seq_len in enumerate(seq_lens):
                                tmp_batch.append(value[idx, :seq_len])
                            batch_output[key].extend(tmp_batch)
                        else:
                            batch_output[key].append(value)
        return batch_output
