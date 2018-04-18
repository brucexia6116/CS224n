#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：分析实体识别结果【错误标注的词】
时间：2017年12月01日14:09:55
"""

from util import read_json


def read_data():
    with open("data/test.json") as f:  # 标准答案
        standard = read_json(f)

    with open("result/test_pred.json") as f:
        result = read_json(f)

    return standard, result


def get_diff_labeled():
    standard, result = read_data()

    record = dict()  # 记录标记不一样的词，在预测中的标记结果
    record2 = []  # 记录标记不一样的词，去重

    for d1, d2 in zip(standard, result):
        for i in range(len(d2[0])):  # 注意，预测值可能会少，被最大长度截断
            word = d1[0][i]
            if d1[1][i] != d2[1][i]:
                temp = " ".join([word, d1[1][i], d2[1][i]])
                if temp not in record2:
                    print("【{}】：{} —— {}".format(word, d1[1][i], d2[1][i]))
                    record2.append(temp)

            if word not in record:
                record[d1[0][i]] = set(d2[1][i])
            else:
                record[d1[0][i]].add(d2[1][i])

    return record


def words_new_finded():
    standard, result = read_data()
    all_words_stan = set()
    for d in standard:
        for word, label in zip(d[0], d[1]):
            if label != "O":
                all_words_stan.add(word)

    all_words_resu = set()
    for d in result:
        for word, label in zip(d[0], d[1]):
            if label != "O":
                all_words_resu.add(word)

    record = get_diff_labeled()
    for w in all_words_resu:
        if w not in all_words_stan:
            print("【{}】".format(w), record[w])  # 新词


if __name__ == "__main__":
    words_new_finded()
