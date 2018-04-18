#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：tensorflow LSTM 用于命名实体识别
时间：2017年09月11日22:45:57
"""

import time
import tensorflow as tf
import numpy as np
import logging
from util import minibatches, load_word_vector_mapping, progress_bar, write_txt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from data_util import load_data, normalize

logger = logging.getLogger("lstm")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# 参数
n_word_features = 1  # 词本身的特征数
window_size = 2
n_features = (2 * window_size + 1) * n_word_features  # 词在一个窗口范围内的feature数
max_length = 200  # 最大长度
embed_size = 50  # 词向量维度
hidden_size = 300  # 隐藏层大小
batch_size = 128  # 批次大小（256太大）
n_epochs = 5  # 轮次
dropout = 0.5
learning_rate = 0.001

output_path = "result"  # 保存结果路径
model_output = "model/model.weights"  # 保存模型（train时）

# 更换语料时，可能需要更改的
n_classes = 9  # label类别个数
LBLS = ["B", "S", "D", "F", "T", "M", "C", "E", "O", ]  # 类别标签
# LBLS = ["PER", "LOC", "ORG", "MISC", "O", ]

vocab_file = "data/vocab.txt"
word2vec_file = "data/word2vec.txt"


def add_placeholders():
    """生成输入张量的占位符
    """
    input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, max_length, n_features))
    labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, max_length))
    mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, max_length))  # 对应句子原始长度的数据

    return input_placeholder, labels_placeholder, mask_placeholder


def create_feed_dict(input_placeholder, mask_placeholder, labels_placeholder,
                     inputs_batch, mask_batch, labels_batch=None):
    """生成feed_dict
    """
    feed_dict = {
        input_placeholder: inputs_batch,
        mask_placeholder: mask_batch,
    }
    if labels_batch is not None:
        feed_dict[labels_placeholder] = labels_batch
    return feed_dict


def add_embedding(helper, input_placeholder):
    """添加embedding层，shape (None, max_length, n_features*embed_size)
    :return:
    """

    vocab = open(vocab_file)
    vectors = open(word2vec_file)
    pre_embeddings = np.array(np.random.randn(
        len(helper.tok2id) + 1, embed_size), dtype=np.float32)

    pre_embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(vocab, vectors).items():
        word = normalize(word)
        if word in helper.tok2id:
            pre_embeddings[helper.tok2id[word]] = vec

    logger.info("初始化 embeddings.")
    vocab.close()
    vectors.close()

    embed = tf.Variable(pre_embeddings, name="embed")
    # shape(None, max_length, n_features)
    features = tf.nn.embedding_lookup(embed, input_placeholder)
    embeddings = tf.reshape(
        features, shape=(-1, max_length, n_features * embed_size))

    return embeddings


def inference_op(helper, input_placeholder):
    """推断模型
    Returns:
        预测值pred: shape (batch_size, max_length, n_classes)，一个batch的预测值
    """

    preds = []  # 预测值
    embeddings = add_embedding(helper, input_placeholder)

    # 输入是一个window的词的词向量的组合
    cell = tf.contrib.rnn.LSTMCell(hidden_size)
    u = tf.get_variable("u", shape=(hidden_size, n_classes),  # 300*3
                        initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", shape=(n_classes,),
                         initializer=tf.constant_initializer(0.0))
    state = cell.zero_state(tf.shape(embeddings)[0], tf.float32)  # 初始化state
    # c = tf.zeros(shape=(tf.shape(x)[0], hidden_size))

    with tf.variable_scope("LSTM"):
        for time_step in range(max_length):  # 每个词
            if time_step != 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(embeddings[:, time_step, :], state)
            o_drop_t = tf.nn.dropout(output, dropout)
            y_t = tf.matmul(o_drop_t, u) + tf.expand_dims(b2, 0)  # 升维，[[0.1,0.4,0.2,0.1,0.1]]

            preds.append(y_t)

    preds = tf.stack(preds, axis=1)  # reshape preds

    return preds  # [p1, p2, p3,] 一个batch的句子，200个词，每个词对应一个3维数组（类别数），每个数是概率


def evaluate(sess, examples_set, examples_raw, examples_vec, pred,
             input_placeholder, mask_placeholder, labels_placeholder):
    """在@examples_raw数据上评估模型/预测数据类别
    """

    preds = []  # 所有数据的预测值存放
    for j, batch in enumerate(minibatches(examples_set, batch_size, shuffle=False)):
        inputs_batch, mask_batch = batch[0], batch[2]
        feed = create_feed_dict(input_placeholder, mask_placeholder, labels_placeholder,
                                inputs_batch=inputs_batch,
                                mask_batch=mask_batch)
        preds_ = sess.run(tf.argmax(pred, axis=2), feed_dict=feed)  # 一个batch的预测值

        preds += list(preds_)

        total_batch = 1 + int(len(examples_set) / batch_size)
        print(progress_bar(j, total_batch, "batch"))

    all_original_labels = []  # 标准答案
    all_predicted_labesl = []  # 预测值

    for i, (sentence, labels) in enumerate(examples_vec):
        _, _, mask = examples_set[i]  # 获取每个句子的mask
        labels_ = [l for l, m in zip(preds[i], mask) if m]  # mask作用（预测值只保留mask标记为True的）

        if len(labels_) == len(labels):  # 最后一个batch
            all_original_labels += labels
            all_predicted_labesl += labels_

    cm = confusion_matrix(all_original_labels, all_predicted_labesl)  # 混淆矩阵
    acc_sorce = accuracy_score(all_original_labels, all_predicted_labesl)
    f_score = f1_score(all_original_labels, all_predicted_labesl, average="micro")
    report = classification_report(all_original_labels, all_predicted_labesl, target_names=LBLS)

    print("准确率：", acc_sorce)
    print("F值：", f_score)
    print("混淆矩阵：\n", cm)
    print("分类结果：\n", report)

    result = []
    for i, (sentence, labels) in enumerate(examples_raw):
        _, _, mask = examples_set[i]  # 获取每个句子的mask
        labels_ = [l for l, m in zip(preds[i], mask) if m]  # mask作用（预测值只保留mask标记为True的）
        orig_labels = [LBLS[l] for l in labels_]  # 将数字标签转回字符表示
        result.append((sentence, orig_labels))

    return result


def model():
    all_data = load_data()
    helper = all_data[0]  # helper[0]，即tok2id——字典（词——id），helper[1]——训练集里的最大句子长度
    train_raw, dev_raw, test_raw = all_data[1]  # 原始字符表示[([词1,词2...],[标签1,标签2...]),()]
    train_vec, dev_vec, test_vec = all_data[2]  # 字符转数字[([[id1], [id2]...], [0,1,0,2,3...]),()]
    train_set, dev_set, test_set = all_data[3]  # 每个词取窗口内的词，每个句子padding为定长

    with tf.Graph().as_default():
        logger.info("建立模型...")

        # 生成placeholders
        input_placeholder, labels_placeholder, mask_placeholder = add_placeholders()

        # 推断，得到训练预测值
        pred = inference_op(helper, input_placeholder)

        # saver
        saver = tf.train.Saver()  # 这个是模型的存储

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)  # 初始化

            saver.restore(sess, model_output)  # 加载模型
            print("模型已加载...")

            start_time = time.time()

            # 在测试集上预测
            result = evaluate(sess, test_set, test_raw, test_vec, pred,
                              input_placeholder, mask_placeholder, labels_placeholder)

            print("预测耗时：{}".format(time.time() - start_time))
            with open("result/test_pred.json", "w+") as f:
                write_txt(f, result)


if __name__ == "__main__":
    model()
