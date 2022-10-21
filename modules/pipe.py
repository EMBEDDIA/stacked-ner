# -*- coding: utf-8 -*-
"""Partially https://github.com/fastnlp/fastNLP"""
import functools
__all__ = [
    "Loader"
]

from fastNLP.io import Pipe
from fastNLP.io import DataBundle
from fastNLP.io.pipe.utils import _add_words_field
from fastNLP.io.utils import check_loader_paths

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP import Const
from typing import Union, Dict

from tqdm import tqdm

from fastNLP.io.file_utils import _get_dataset_url, get_cache_path, cached_path


def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:

        sample = []
        start = next(f).strip()

        data = []
        if start != '':
            sample.append(
                start.split(sep)) if sep else sample.append(
                start.split())

        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if 'DOCSTART' in line:
                continue
            if '###' in line:
                continue
            if "# id" in line:
                continue
            if "Token" in line:
                continue
            if "TOKEN" in line:
                continue
            if '###' in line:
                continue
            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                            continue
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:

                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        data.append([line_idx, res])
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


def iob2bioes(tags):

    new_tags = []
#    print(tags)
    for i, tag in enumerate(tags):
        # print(tag)
        tag = tag.replace('S-', 'B-')
        tag = tag.replace('E-', 'I-')
        #tag = tag.replace('B-', 'I-')
        if '-' not in tag:
            if tag != 'O':
                tag = 'O'
        if '0' in tag:
            print('WTF')
        if tag == 'O':
            new_tags.append(tag)
        else:
            split = tag.split('-')[0]
            if split == 'B':
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif split == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise TypeError("Invalid IOB format.")
    return new_tags


def iob2(tags):

    for i, tag in enumerate(tags):
        if '0' in tag:
            tag = 'O'
        tag = tag.replace('S-', 'B-')
        tag = tag.replace('E-', 'I-')
        tag = tag.replace('B-', 'I-')
        if '-' not in tag:
            if tag != 'O':
                tag = 'O'

        if tag == "O":
            continue
        if '-' not in tag:
            if tag != 'O':
                tag = 'O'

        split = tag.split("-")
        if '口' in tag:
            split = 'O'
        if len(split) != 2 or split[0] not in ["I", "B"]:
            print(split)
            raise TypeError("The encoding schema is not a valid IOB type.")
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return tags


def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class Loader:

    def __init__(self):
        pass

    def _load(self, path: str) -> DataSet:

        raise NotImplementedError

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:

        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)

        datasets = {}
        for name, path in tqdm(paths.items(), total=len(paths.items())):
            datasets[name] = self._load(path)
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

    def download(self) -> str:

        raise NotImplementedError(
            f"{self.__class__} cannot download data automatically.")

    @staticmethod
    def _get_dataset_path(dataset_name):

        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(
            url_or_filename=url, cache_dir=default_cache_path, name='dataset')

        return output_dir


def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
            matches.append((pattern, i))
    return matches


class NERLoader(Loader):

    def __init__(self, sep=' ', dropna=True):
        super(NERLoader, self).__init__()
        headers = [
            'raw_words', 'target'
        ]
        # TODO: This needs to be changed if the data format is different or the
        # order of the elements in the file is different
        indexes = [0, -1]
        if not isinstance(headers, (list, tuple)):
            raise TypeError(
                'invalid headers: {}, should be list of strings'.format(headers))
        self.headers = headers
        self.dropna = dropna
        self.sep = sep
        if indexes is None:
            self.indexes = list(range(len(self.headers)))
        else:
            if len(indexes) != len(headers):
                raise ValueError
            self.indexes = indexes

    @functools.lru_cache(maxsize=1024)
    def _load(self, path):
        ds = DataSet()
        idx = 0
        for element in _read_conll(
                path,
                indexes=self.indexes,
                dropna=self.dropna):
            idx, data = element
            # if data[0][0] == '#' and data[1][0] == '--':
            #     continue
            # if data[0][0] == '#':
            #     data[0] = data[0][1:]
            #     data[1] = data[1][1:]
            if len(data[0]) < 1:
                continue
            for i in range(len(self.headers)):
                try:
                    if data[i][0].startswith('NE-'):
                        data[i] = data[i][1:]
                except BaseException:
                    raise TypeError("Data problem.")
                if 'TOKEN' in data[i][0].upper():
                    data[i] = data[i][1:]

                if len(data[i]) == 0:
                    continue
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if len(field) == 0:
                    break
                # if str(' '.join(list(field))).startswith(' #'):
                    # continue
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
                # if str(' '.join(list(field))).startswith('#'):
                #     continue

            if doc_start:
                continue

            idx += 1
            ins = {h: data[i] for i, h in enumerate(self.headers)}

            ds.append(Instance(**ins))
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds

    def download(self):
        raise RuntimeError("conll2003 cannot be downloaded automatically.")


def word_shape(words):
    shapes = []
    for word in words:
        caps = []
        for char in word:
            caps.append(char.isupper())
        if all(caps):
            shapes.append(0)
        elif any(caps) is False:
            shapes.append(1)
        elif caps[0]:
            shapes.append(2)
        elif any(caps):
            shapes.append(3)
        else:
            shapes.append(4)
    return shapes


def _indexize(
        data_bundle,
        input_field_names=Const.INPUT,
        target_field_names=Const.TARGET,
        vocabulary=None):

    if isinstance(input_field_names, str):
        input_field_names = [input_field_names]
    if isinstance(target_field_names, str):
        target_field_names = [target_field_names]

    for input_field_name in input_field_names:
        if vocabulary is None:
            src_vocab = Vocabulary()

            src_vocab.from_dataset(
                *
                [
                    ds for name,
                    ds in data_bundle.iter_datasets() if 'train' in name],
                field_name=input_field_name,
                no_create_entry_dataset=[
                    ds for name,
                    ds in data_bundle.iter_datasets() if (
                        'train' not in name) and (
                        ds.has_field(input_field_name))])

        else:
            src_vocab = vocabulary

        src_vocab.index_dataset(
            *data_bundle.datasets.values(), field_name=input_field_name)

        data_bundle.set_vocab(src_vocab, input_field_name)
    for target_field_name in target_field_names:
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(
            *
            [
                ds for name,
                ds in data_bundle.iter_datasets() if 'train' in name],
            field_name=target_field_name,
            no_create_entry_dataset=[
                ds for name,
                ds in data_bundle.iter_datasets() if (
                    'train' not in name) and (
                    ds.has_field(target_field_name))])
        if len(tgt_vocab._no_create_word) > 0:
            warn_msg = f"There are {len(tgt_vocab._no_create_word)} `{target_field_name}` labels" \
                       f" in {[name for name in data_bundle.datasets.keys() if 'train' not in name]} " \
                       f"data set but not in train data set!.\n" \
                       f"These label(s) are {tgt_vocab._no_create_word}"
            print(warn_msg)
        tgt_vocab.index_dataset(*[ds for ds in data_bundle.datasets.values()
                                if ds.has_field(target_field_name)], field_name=target_field_name)
        data_bundle.set_vocab(tgt_vocab, target_field_name)

    return data_bundle


class DataReader(Pipe):

    def __init__(
            self,
            encoding_type: str = 'bio',
            lower: bool = False,
            word_shape: bool = False,
            vocabulary=None):

        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape
        self.vocabulary = vocabulary

    def process(self, data_bundle: DataBundle) -> DataBundle:
        #import pdb;pdb.set_trace()
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(
                self.convert_tag,
                field_name=Const.TARGET,
                new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(
                word_shape,
                field_name='raw_words',
                new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')
        data_bundle.apply_field(
            lambda chars: [
                ''.join(
                    [
                        '0' if c.isdigit() else c for c in char]) for char in chars],
            field_name=Const.INPUT,
            new_field_name=Const.INPUT)

        _indexize(
            data_bundle,
            input_field_names=[
                Const.INPUT],
            target_field_names=['target'],
            vocabulary=self.vocabulary)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:

        data_bundle = NERLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle
