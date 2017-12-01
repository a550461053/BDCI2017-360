#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import random
from gensim.models import Word2Vec

# ==================================================
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("input_text_file", "./data/train_and_test/train_test_seg", "Test text data source to evaluate.")
tf.flags.DEFINE_string("input_label_file", "./data/train_and_test/train_test_label", "Label file for test text data source.")
# tf.flags.DEFINE_string("positive_data_file", './data/train_and_test/train_pos', "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", './data/train_and_test/train_neg', "Data source for the negative data.")

# Eval Parameters
# 1511026951 新数据0.8
# 1510976108 小数据

# 1511932589 - 2000 len
# 1511884581 - 1000 len
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)") # 1508153975 1507963188
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1511884581/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)") # 词向量维度

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# ./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# ==================================================
# validate
# ==================================================
# 1. validate checkout point file
import word2vec_helpers
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# 3. validate training params file
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# 4. Load params
params = data_helpers.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])

# 5. Load data
if FLAGS.eval_train: # 使用训练测试集
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.input_text_file, FLAGS.input_label_file, num_labels)
else: # 使用真正的测试集
    # x_raw = ["a masterpiece four years in the making", "everything is off."]
    x_raw = '../../BDCI2017-360/BDCI2017-360-Semi/evaluation_public_seg.tsv'#evaluation_public_seg.tsv'

    # x_raw 是需要经过分词处理的sentences
import re
## load data
eval_data = []
with open(x_raw, 'r', encoding='UTF-8') as file:
    lines = file.readlines() # 速度快
    count = 0
    for line1 in lines:
        # count += 1
        # if count <= 499500:
        #     continue
        str_list = line1.strip('\n').split('\t')
        line = str_list
        id_, title_, str_ = line[0], line[1], line[2]
        title_ = [item.lower() for item in title_.split(' ') if item != ' ']
        str_ = [item.lower() for item in str_.split(' ') if item != ' ']
        # merge to string
        title_ = ' '.join(title_)
        str_ = ' '.join(str_)
        # ②只保留中文字符，并且保留分割效果
        title_clean = ' '.join(re.findall(r'[\u4e00-\u9FA5a-zA-Z，。,.]+', title_))
        str_clean = ' '.join(re.findall(r'[\u4e00-\u9FA5a-zA-Z，。,.]+', str_))
        res = [id_, title_clean, str_clean]
        # print('clean:', res)
        eval_data.append(res)
print('eval data length :', len(eval_data))


### define eval step
def eval_batch_iter_my(eval_data, file_to_load, batch_size, shuffle=False):
    # 1.embedding
    print('\t load the file:', file_to_load)
    w2vModel = Word2Vec.load(file_to_load)

    embeddingDim = w2vModel.vector_size
    embeddingUnknow = [0 for i in range(embeddingDim)]

    data_size = len(eval_data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    print('num_batches_per_epoch:', num_batches_per_epoch)
    print('eval_data_size:', data_size)

    for batch_num in range(num_batches_per_epoch):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, data_size)
        x_vector = []
        for i in range(start_idx, end_idx):
            word_count = 0
            sentence = eval_data[i]
            id_ = sentence[0]
            words = (sentence[1] + ' ' + sentence[2]).strip('\n').split(' ')
            for word in words:
                if word in w2vModel.wv.vocab:
                    x_vector.append(w2vModel[word])
                else:
                    x_vector.append(embeddingUnknow)
                word_count += 1
                if word_count >= max_document_length:
                    break
            if word_count < max_document_length:
                for i in range((max_document_length - word_count)):
                    x_vector.append(embeddingUnknow)
        x_test = np.array(x_vector).reshape(-1, max_document_length, FLAGS.embedding_dim, 1)
        yield x_test

# ==================================================
# Evaluation
# ==================================================
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        # print(type(x_test), x_test)
        # 直接测试就不需要shuffle了
        batches = eval_batch_iter_my(eval_data, '../../BDCI2017-360/model/word2vec_semi_final.model',
                                     FLAGS.batch_size,
                                     shuffle=True)  # x_train比较大，作为一个迭代器
        # Collect the predictions here
        all_predictions = []
        for x_test_batch in batches:
            # print(len(x_test_batch))
            # print(np.shape(x_test_batch))
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            print('\t batch predict is:', batch_predictions)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        # for i in range(5):
        #     # print(type(x_test_batch))
        #     batch_predictions = sess.run(predictions, {input_x: x_test[i], dropout_keep_prob: 1.0})
        #     # all_predictions = np.concatenate([all_predictions, batch_predictions])
        #     all_predictions.append(batch_predictions)

# Print accuracy if y_test is defined
# if y_test is not None:
#     print(type(y_test), y_test)
#     y_test_ = []
#     for i in y_test:
#         y_test_.append(int(i))
#     y_test = y_test_
#     correct_predictions = float(sum(np.array(all_predictions) == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


# Save the evaluation to a csv
with open('result', 'w', encoding='UTF-8') as file:
    for item in all_predictions:
        # print(item)
        file.write(str(item)+'\n')

# predictions_human_readable = np.column_stack((np.array([text for text in x_raw]), all_predictions)) # 不需要再次：text.encode('utf-8') 了
# # predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w', encoding='UTF-8') as f:
#     csv.writer(f).writerows(predictions_human_readable)
