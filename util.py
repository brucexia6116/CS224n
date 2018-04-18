#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util.py: General utility routines
"""

import re
import time
import logging
import json
import pickle
from collections import OrderedDict
import numpy as np

logger = logging.getLogger("lstm")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

LBLS = ["PER", "LOC", "ORG", "MISC", "O", ]


def progress_bar(current_step, all_steps, annotation=None):
    current_step += 1
    j = (current_step * 20) // all_steps
    s1 = "[[{}{}]  {:.2f}%]".format(">" * j, "." * (20 - j), current_step * 100 / all_steps)
    s2 = "\r{}：{}/{}  {}".format(annotation, current_step, all_steps, s1)
    time.sleep(1)

    return s2


def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    给定词列表vocab和向量列表vector，将两者一一对应起来；
    前提是两者本身是顺序对应的。
    """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = np.array(list(map(float, vector.split())))

    return ret


def window_iterator(seq, n=1, beg="<s>", end="</s>"):
    """遍历
    Iterates through seq by returning windows of length 2n+1
    """
    for i in range(len(seq)):
        l = max(0, i - n)
        r = min(len(seq), i + n + 1)
        ret = seq[l:r]
        if i < n:
            ret = [beg, ] * (n - i) + ret
        if i + n + 1 > len(seq):
            ret = ret + [end, ] * (i + n + 1 - len(seq))
        yield ret


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    """
    list_data = type(data) is list and (
        type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)  # 事实上没有错误
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:
        minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


def del_space(matched):
    mod = matched.group('value').replace(" ", "")  # 去除实体和标签之间的空格
    return mod


def read_json(fstream, u=None):
    """
    列表里是所有句子，每个句子表示为集合（[词1, 词2, ... , 词n], [标签1， 标签2, ... 标签n]）；
    词和标签均为字符串。
    :param fstream:
    :param u:
    :return:
    """

    ret = []
    number_record = []

    all_data = json.load(fstream)
    for data in all_data:

        number_record.append([data["doc_id"], data["line_num"]])  # 保存编号
        sentence = data["sentence"]  # 读取句子（已分词和标注实体）

        for l in LBLS:
            sentence = re.sub('(?P<value><({})>(.*?)</{}>)'.format(l, l), del_space, sentence).strip()

        pat2 = re.compile("<(.*?)>(.*?)</.*?>")
        words = []
        labels = []
        for word in sentence.split(" "):
            finds = re.findall(pat2, word)  # 判断是否是实体
            if finds:
                words.append(finds[0][1])  # 实体词
                labels.append(finds[0][0])  # label词
            else:
                words.append(word)  # 非实体词
                labels.append("O")  # 标签-O

        ret.append((words, labels))

    if u:
        output = open('data/number_record.pkl', 'wb')  # 注意，多次调用会覆盖
        pickle.dump(number_record, output)  # 编号存入output文件
        output.close()

    return ret


def write_txt(fstream, data):
    """
    :param fstream:
    :param data:要写入的数据，[([]，[]),] 格式
    :return:
    """
    sen_list = []
    for item in data:
        single_sen = []
        p_words, p_labels = item[0], item[1]

        for word, label in zip(p_words, p_labels):
            if label != "O":
                nw = "<{}> {} </{}>".format(label, word, label)
                single_sen.append(nw)
            else:
                single_sen.append(word)
        sen_list.append(single_sen)

    f_num = open("data/number_record.pkl", "rb")
    number_record = pickle.load(f_num)
    f_num.close()

    all_data = []

    for sen, sen_num in zip(sen_list, number_record):
        temp_dic = dict()
        temp_dic["doc_id"] = sen_num[0]
        temp_dic["line_num"] = sen_num[1]
        temp_dic["sentence"] = " ".join(sen)
        all_data.append(temp_dic)

    json.dump(all_data, fstream, ensure_ascii=False, indent=4)
