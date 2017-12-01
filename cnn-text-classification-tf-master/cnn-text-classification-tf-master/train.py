#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import random
from gensim.models import Word2Vec

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .001, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../../BDCI2017-360/BDCI2017-360-Semi/train_seg.tsv", "Data source for the train_seg data.")
tf.flags.DEFINE_string("test_data_file", "../../BDCI2017-360/BDCI2017-360-Semi/evaluation_public_seg.tsv", "Data source for the test_seg data.")
tf.flags.DEFINE_string("train_data_old_file", "../../BDCI2017-360/train_seg.tsv", "Data source for the train_seg data.")
tf.flags.DEFINE_string("test_data_old_file", "../../BDCI2017-360/evaluation_public_seg.tsv", "Data source for the test_seg data.")


tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")
tf.flags.DEFINE_integer("sentence_length", 2000, "sentence_length")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)") # 词向量维度
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')") # 3,4,5即为分别划过3,4,5个字，总共有3*num_filters个过滤器
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)") # 卷积核数目
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.flags.DEFINE_float("decay_rate", 0.6, "decay_rate (default:0.98)")
# decay_step 默认为每个epoch改变一次
# semi final - per epoch 4683 steps
# gpu：200 len, 12485s
# gpu：0.001 1000 len, 46655s


# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)") # 每批训练大小
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)") # 总迭代次数
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("word2vec_model", '../../BDCI2017-360/model/word2vec_semi_final.model', "the model of word2vec_model (default: '../../BDCI2017-360/model/word2vec_test.model')")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices") # tf会登录CPU或GPU进行操作


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\n总体参数Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# ==================================================
# 一、 Prepare output directory for models and summaries
# ==================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# ==================================================
# 二、 Data Preparation
# ==================================================
# 1. Load data and shuffle data
print("---------- Loading data -----------")
train_data_all = data_helpers.load_train_data_file(FLAGS.train_data_file, clean=True)
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(train_data_all)))
train_data = train_data_all[:dev_sample_index]
test_data = train_data_all[dev_sample_index:]
print('train data:', len(train_data), 'test data:', len(test_data))

# 2. Save params
max_document_length = FLAGS.sentence_length
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels': FLAGS.num_labels, 'max_document_length': max_document_length}
data_helpers.saveDict(params, training_params_file)

# 3. define train step
def train_batch_iter_my(train_data, file_to_load, batch_size,
                        num_epoches, shuffle=False):
    # 1.embedding
    print('\t load the file:', file_to_load)
    w2vModel = Word2Vec.load(file_to_load)

    embeddingDim = w2vModel.vector_size
    embeddingUnknow = [0 for i in range(embeddingDim)]

    data_size = len(train_data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    print('num_batches_per_epoch:', num_batches_per_epoch)
    print('data_size:', data_size)
    print('num_epoches:', num_epoches)

    for epoch in range(num_epoches):
        print('epoch:', epoch)
        if shuffle:
            # random.seed(21)
            random.shuffle(train_data)
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            x_vector = []
            y_label = []
            for i in range(start_idx, end_idx):
                # print(train_data[i][0][0], train_data[i][1])

                word_count = 0
                sentence = train_data[i][0]
                y_label.append(train_data[i][1])
                id_ = sentence[0]
                words = (sentence[1] + sentence[2]).strip('\n').split(' ')
                for word in words:
                    if word in w2vModel.wv.vocab:
                        x_vector.append(w2vModel[word])
                    else:
                        x_vector.append(embeddingUnknow)
                    word_count += 1
                    if word_count >= FLAGS.sentence_length:
                        break
                if word_count < FLAGS.sentence_length:
                    for i in range((FLAGS.sentence_length - word_count)):
                        x_vector.append(embeddingUnknow)
                # print(sentence[0], y_label)
                # input()
            x_train = np.array(x_vector).reshape(-1, FLAGS.sentence_length, FLAGS.embedding_dim, 1)
            y_train = np.array(y_label).reshape(-1, 2)
            yield x_train, y_train

# 4. define train step
def test_batch_iter_my(test_data, file_to_load, batch_size,
                        num_epoches, shuffle=False):
    # 1.embedding
    print('\t load the file:', file_to_load)
    w2vModel = Word2Vec.load(file_to_load)

    embeddingDim = w2vModel.vector_size
    embeddingUnknow = [0 for i in range(embeddingDim)]

    data_size = len(test_data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    print('num_batches_per_epoch:', num_batches_per_epoch)
    print('data_size:', data_size)
    print('num_epoches:', num_epoches)

    for batch_num in range(num_batches_per_epoch):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, data_size)
        x_vector = []
        y_label = []
        for i in range(start_idx, end_idx):
            word_count = 0
            sentence = test_data[i][0]
            y_label.append(test_data[i][1])
            id_ = sentence[0]
            words = (sentence[1] + sentence[2]).strip('\n').split(' ')
            for word in words:
                if word in w2vModel.wv.vocab:
                    x_vector.append(w2vModel[word])
                else:
                    x_vector.append(embeddingUnknow)
                word_count += 1
                if word_count >= FLAGS.sentence_length:
                    break
            if word_count < FLAGS.sentence_length:
                for i in range((FLAGS.sentence_length - word_count)):
                    x_vector.append(embeddingUnknow)
        x_test = np.array(x_vector).reshape(-1, FLAGS.sentence_length, FLAGS.embedding_dim, 1)
        y_test = np.array(y_label).reshape(-1, 2)
        yield x_test, y_test

# ==================================================
# 三、 Training
# ==================================================
print('start training...')
with tf.Graph().as_default():
    with tf.device("/gpu:0"): # GPU分配内存不够，需要2.08G。。。，改为cpu即可。
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=FLAGS.sentence_length,
                num_classes=2,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # learning_rate = FLAGS.learning_rate
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                       decay_steps=len(train_data)/FLAGS.batch_size,
                                                       decay_rate=FLAGS.decay_rate, #0.1,
                                                       staircase=True)
            # decay_steps表示经过多少次step，进行学习率调整
            # 参数staircase如果设置为True，那么指数部分就会采用整除策略，表示每decay_step，学习速率变为原来的decay_rate，至于第四个参数decay_rate表示的是学习速率的下降倍率
            # auto learning rate
            print('decay_steps = ', len(train_data) / FLAGS.batch_size)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # 学习率太低了会过拟合
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy 跟踪参数总结
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train_and_test")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # use tensorboard cmd: tensorboard --logdir=train_and_test

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it 保存模型的参数以便稍后恢复
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            # batches = data_helpers.batch_iter(
            #     list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            batches = train_batch_iter_my(
                train_data, FLAGS.word2vec_model, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)  # x_train比较大，作为一个迭代器

            # Training loop. For each batch...
            start_time = time.time()
            for batch in batches:
                x_batch, y_batch = batch
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                print('the learning rate is:', learning_rate.eval(session=sess))
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    for x_dev, y_dev in test_batch_iter_my(test_data, FLAGS.word2vec_model,
                                                           FLAGS.batch_size, FLAGS.num_epochs, shuffle=True):  # x_train比较大，作为一个迭代器
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            stop_time = time.time()
            print('cost time:', stop_time-start_time)
            # 学习率为0.05，耗时：7273.915070295334

