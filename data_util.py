#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：处理数据
时间：2017年09月14日15:50:29
"""
import os
import pickle
import logging
from collections import Counter

from util import read_json, window_iterator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

START_TOKEN = "<s>"
END_TOKEN = "</s>"

LBLS = ["PER", "LOC", "ORG", "MISC", "O", ]
NONE = "O"
NUM = "NNNUMMM"
UNK = "UUUNKKK"

n_word_features = 1  # 词本身的特征数
window_size = 2
n_features = (2 * window_size + 1) * n_word_features  # 词在一个窗口范围内的feature数
max_length = 200  # 最大长度
embed_size = 50  # 词向量的维度

train_file_path = "data/train.conll"
dev_file_path = "data/dev.conll"
test_file_path = "data/test.conll"


def read_conll(fstream):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
    """
    ret = []

    current_toks, current_lbls = [], []
    for line in fstream:
        line = line.strip()
        if len(line) == 0 or line.startswith("-DOCSTART-"):
            if len(current_toks) > 0:
                assert len(current_toks) == len(current_lbls)
                ret.append((current_toks, current_lbls))
            current_toks, current_lbls = [], []
        else:
            assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
            tok, lbl = line.split("\t")
            current_toks.append(tok)
            current_lbls.append(lbl)
    if len(current_toks) > 0:
        assert len(current_toks) == len(current_lbls)
        ret.append((current_toks, current_lbls))
    return ret


def normalize(word):
    """数字→NUM，大写字母→小写
    """
    if word.isdigit():
        return NUM
    else:
        return word.lower()


def pad_sequences(data):
    """序列pad为最大长度（补零或是截取）
    """
    ret = []
    zero_vector = [0] * n_features  # 零向量维度与词的特征数一致，例如都是2
    zero_label = 1  # "零"的标签；随时随着我的类别的个数变化

    for sentence, labels in data:
        new_sent = sentence + [zero_vector] * max_length  # 新句子补0
        new_labels = labels + [zero_label] * max_length  # 新标签补1
        mask = [True] * len(labels) + [False] * max_length  # mask，补上的对应False
        ret.append(
            (new_sent[:max_length], new_labels[:max_length], mask[:max_length]))  # 截取最大长度

    return ret


class ModelHelper(object):
    """数据预处理、构造embeddings，等.
    """

    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id
        self.START = [tok2id[START_TOKEN]]
        self.END = [tok2id[END_TOKEN]]
        self.max_length = max_length

    def vectorize_example(self, sentence, labels=None):
        # 如果句子中的词在tok2id字典中存在，则返回其编号，否则返回UNK的编号
        # sentence_格式：[[id1], [id2], ..., [idn]]
        # labels_格式：[0, 1, 0, 3, 1, 2...]  数字list
        sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK])] for word in sentence]
        if labels:
            labels_ = [LBLS.index(l) for l in labels]
            return sentence_, labels_
        else:
            return sentence_, [LBLS[-1] for _ in sentence]

    def vectorize(self, data):
        return [self.vectorize_example(sentence, labels) for sentence, labels in data]

    @classmethod
    def build(cls, data):
        """给定数据data，生成一个词语——编号id对应的字典
        :param data:数据格式[([词1, 词2, ...], [标签1, 标签2, ...]), (...)]
        """
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        words = [normalize(word) for sentence, _ in data for word in sentence]  # 语料所有词
        tok2id = build_dict(words, offset=1, max_words=10000)  # 建立词的编号id字典

        others = [START_TOKEN, END_TOKEN, UNK]  # 句子开始、结束、未知词UNK
        tok2id.update(build_dict(others, offset=(len(tok2id) + 1)))

        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1  # 确定编号从1开始
        logger.info("建立语料词典，共 %d 词。", len(tok2id))

        max_length = max(len(sentence) for sentence, _ in data)  # 所有句子里的最大长度

        return cls(tok2id, max_length)

    def save(self, path):
        # 保存 tok2id map.
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "features.pkl"), "wb") as f:
            pickle.dump([self.tok2id, self.max_length], f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "features.pkl"), 'rb') as f:
            tok2id, setence_max_length = pickle.load(f)
        return cls(tok2id, setence_max_length)


def load_data():
    # raw原始数据
    with open(train_file_path) as data_train:
        train_r = read_conll(data_train)
    logger.info("训练集加载完成，共有 %d 句", len(train_r))

    with open(dev_file_path) as data_dev:
        dev_r = read_conll(data_dev)
    logger.info("验证集加载完成，共有 %d 句", len(dev_r))

    with open(test_file_path) as data_test:
        test_r = read_conll(data_test)  # 保存编号
    logger.info("测试集加载完成，共有 %d 句", len(test_r))

    helper = ModelHelper.build(train_r)  # helper

    # vec，处理后的数据
    train_v = helper.vectorize(train_r)
    dev_v = helper.vectorize(dev_r)
    test_v = helper.vectorize(test_r)

    # set，padding后的数据
    train_s = pad_sequences(featurize_windows(train_v, helper.START, helper.END))
    dev_s = pad_sequences(featurize_windows(dev_v, helper.START, helper.END))
    test_s = pad_sequences(featurize_windows(test_v, helper.START, helper.END))

    return helper, (train_r, dev_r, test_r), (train_v, dev_v, test_v), (train_s, dev_s, test_s)


def featurize_windows(data, start, end):
    """窗口取词，每个句子前后加<s>，</s>的编号，保证窗口长度统一
    """
    ret = []
    for sentence, labels in data:  # 此时都是数字编号

        sentence_ = []
        for window in window_iterator(sentence, window_size, beg=start, end=end):
            sentence_.append(sum(window, []))
        ret.append((sentence_, labels))
    return ret


def build_dict(words, max_words=None, offset=0):
    """
    建立了一个字典，这个字典里包含了这个单词和词频
    """
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset + i for i, (word, _) in enumerate(words)}


if __name__ == '__main__':
    load_data()
