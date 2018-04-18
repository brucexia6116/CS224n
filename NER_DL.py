#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：基于斯坦福CS224n作业3的命名实体识别
时间：2018年04月18日11:36:05
"""
import time
import tensorflow as tf
import numpy as np
import logging
from util import minibatches, load_word_vector_mapping, progress_bar
from data_util import load_data, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

logger = logging.getLogger("NER_DL")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class NER_DL:
    """深度学习命名实体识别"""

    def __init__(self):
        # logging
        self.logger = logging.getLogger("NER_model")

        # 参数
        self.n_word_features = 1  # 词本身的特征数
        self.window_size = 2
        self.n_features = (2 * self.window_size + 1) * self.n_word_features  # 词在一个窗口范围内的feature数
        self.max_length = 200  # 最大长度
        self.embed_size = 50  # 词向量维度
        self.hidden_size = 300  # 隐藏层大小
        self.batch_size = 32  # 批次大小（256太大）
        self.n_epochs = 1  # 轮次
        self.dropout = 0.5
        self.learning_rate = 0.001

        # 更换语料时，可能需要更改的
        self.n_classes = 5  # label类别个数
        self.LBLS = ["PER", "ORG", "LOC", "MISC", "O"]  # 类别标签

        self.vocab_file = "data/vocab.txt"
        self.word2vec_file = "data/word2vec.txt"

        self.output_path = "results"  # 保存结果路径
        self.model_output = "model/model.weights"  # 保存模型（train时）
        self.log_dir = 'logs'

        all_data = load_data()
        self.helper = all_data[0]  # helper[0]，即tok2id——字典（词——id），helper[1]——训练集里的最大句子长度
        # train_raw, dev_raw, test_raw = all_data[1]  # 原始字符表示[([词1,词2...],[标签1,标签2...]),()]
        self.train_vec, self.dev_vec, self.test_vec = all_data[2]  # 字符转数字[([[id1], [id2]...], [0,1,0,2,3...]),()]
        self.train_set, self.dev_set, self.test_set = all_data[3]  # 每个词取窗口内的词，每个句子padding为定长
        self.helper.save(self.output_path)  # 存储为features.pkl，词典

        self.model()

    def add_placeholders(self):
        """生成输入张量的占位符
        """
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.max_length, self.n_features))
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.max_length))
        self.mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_length))  # 对应句子原始长度的数据
        self.dropout_placeholder = tf.placeholder(tf.float32, name="drop")

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1.0):
        """生成feed_dict
        """
        if labels_batch is None:
            feed_dict = {
                self.input_placeholder: inputs_batch,
                self.mask_placeholder: mask_batch,
                self.dropout_placeholder: dropout}
        else:
            feed_dict = {
                self.input_placeholder: inputs_batch,
                self.mask_placeholder: mask_batch,
                self.labels_placeholder: labels_batch,
                self.dropout_placeholder: dropout}

        return feed_dict

    def add_embedding(self, helper, input_placeholder):
        """添加embedding层，shape (None, max_length, n_features*embed_size)
        :return:
        """

        vocab = open(self.vocab_file)
        vectors = open(self.word2vec_file)
        pre_embeddings = np.array(np.random.randn(
            len(helper.tok2id) + 1, self.embed_size), dtype=np.float32)

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
        self.embeddings = tf.reshape(
            features, shape=(-1, self.max_length, self.n_features * self.embed_size))

        return self.embeddings

    def inference_op(self):
        """推断模型
        Returns:
            预测值pred: shape (batch_size, max_length, n_classes)，一个batch的预测值
        """

        self.preds = []  # 预测值
        embeddings = self.add_embedding(self.helper, self.input_placeholder)

        # 输入是一个window的词的词向量的组合
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
        u = tf.get_variable("u", shape=(self.hidden_size, self.n_classes),  # 300*3
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=(self.n_classes,),
                             initializer=tf.constant_initializer(0.0))
        state = cell.zero_state(tf.shape(embeddings)[0], tf.float32)  # 初始化state
        # c = tf.zeros(shape=(tf.shape(x)[0], hidden_size))

        with tf.variable_scope("LSTM"):
            for time_step in range(self.max_length):  # 每个词
                if time_step != 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(embeddings[:, time_step, :], state)
                o_drop_t = tf.nn.dropout(output, self.dropout)
                y_t = tf.matmul(o_drop_t, u) + tf.expand_dims(b2, 0)  # 升维，[[0.1,0.4,0.2,0.1,0.1]]

                self.preds.append(y_t)

        self.preds = tf.stack(self.preds, axis=1)  # reshape preds

        return self.preds  # [p1, p2, p3,] 一个batch的句子，200个词，每个词对应一个3维数组（类别数），每个数是概率

    def loss_op(self):
        """计算loss
        tf.boolean_mask：将句子长度的（有意义的）cross_entropy选出来（若干个）
        tf.reduce_mean：取平均

        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                                       logits=self.preds)
        self.loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, self.mask_placeholder))
        return self.loss

    def add_training_op(self):
        """训练op
        """
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

    def train_on_epoch(self, sess, summary, summary_writer):
        # 分batch训练，batch里有batch_size个句子、labels、masks
        for i, batch in enumerate(minibatches(self.train_set, self.batch_size)):
            inputs_batch, labels_batch, mask_batch = batch[0], batch[1], batch[2]

            feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                         mask_batch=mask_batch,
                                         labels_batch=labels_batch,
                                         dropout=self.dropout)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)

            if i % 10 == 0:
                summary_str = sess.run(summary, feed_dict=feed)
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()

            total_batch = 1 + int(len(self.train_set) / self.batch_size)
            print(progress_bar(i, total_batch, "batch") + "  train_loss：{:.8f}".format(loss))

    def train_epochs(self, sess, saver, summary, summary_writer):
        best_score = 0.  # 记录n_epoch次中最好的分数

        for epoch in range(self.n_epochs):
            start_time = time.time()
            print(progress_bar(epoch, self.n_epochs, "epoch"))

            self.train_on_epoch(sess, summary, summary_writer)  # 训练模型

            logger.info("【Dev】Evaluating on development data...")
            _, best_score = self.evaluate(sess, best_score, saver)  # 每个epoch的评估

            print("第{}轮耗时：{:.2f} s\n\n".format(epoch + 1, time.time() - start_time))
            print("--------每轮分割线-------\n\n")

        print(best_score)

    def evaluate(self, sess, pre_score, saver=None):
        """在@examples_raw数据上评估模型/预测数据类别
        """
        new_score = pre_score
        preds = []  # 所有数据的预测值存放
        for j, batch in enumerate(minibatches(self.dev_set, self.batch_size, shuffle=False)):
            inputs_batch, mask_batch = batch[0], batch[2]
            feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                         mask_batch=mask_batch)
            preds_ = sess.run(tf.argmax(self.preds, axis=2), feed_dict=feed)  # 一个batch的预测值

            preds += list(preds_)

            total_batch = 1 + int(len(self.dev_set) / self.batch_size)
            print(progress_bar(j, total_batch, "batch"))

        all_original_labels = []  # 标准答案
        all_predicted_labesl = []  # 预测值

        for i, (sentence, labels) in enumerate(self.dev_vec):
            _, _, mask = self.dev_set[i]  # 获取每个句子的mask
            labels_ = [l for l, m in zip(preds[i], mask) if m]  # mask作用（预测值只保留mask标记为True的）

            if len(labels_) == len(labels):  # 最后一个batch
                all_original_labels += labels
                all_predicted_labesl += labels_

        cm = confusion_matrix(all_original_labels, all_predicted_labesl)  # 混淆矩阵
        acc_sorce = accuracy_score(all_original_labels, all_predicted_labesl)
        f_score = f1_score(all_original_labels, all_predicted_labesl, average="micro")
        report = classification_report(all_original_labels, all_predicted_labesl, target_names=self.LBLS)

        print("准确率：", acc_sorce)
        print("F值：", f_score)
        print("混淆矩阵：\n", cm)
        print("分类结果：\n", report)

        if f_score > pre_score:
            new_score = f_score
            if saver:  # 训练时可保存下最好的模型，测试时可选择没有saver
                logger.info("New best score! Saving model in %s", self.model_output)
                saver.save(sess, self.model_output)

        return all_predicted_labesl, new_score

    def model(self):
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Graph().as_default():
            logger.info("建立模型...")

            self.add_placeholders()
            self.inference_op()
            self.loss_op()
            self.add_training_op()
            summary = tf.summary.merge_all()
            saver = tf.train.Saver()  # 这个是模型的存储

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)  # 初始化
                summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)  # summary

                # 在训练集上训练，每个epoch在验证集上评估
                self.train_epochs(sess, saver, summary, summary_writer)


NER_model = NER_DL()

if __name__ == "__main__":
    pass
