# coding=utf-8
import numpy as np
import re
import itertools
from collections import Counter
import os
import word2vec_helpers
import time
import pickle
from gensim.models import Word2Vec
import multiprocessing

## 原来
# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()
#
#
# def load_data_and_labels(positive_data_file, negative_data_file):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     positive_examples = list(open(positive_data_file, "r", encoding='UTF-8').readlines())
#     positive_examples = [s.strip() for s in positive_examples]
#     negative_examples = list(open(negative_data_file, "r", encoding='UTF-8').readlines())
#     negative_examples = [s.strip() for s in negative_examples]
#     # Split by words
#     x_text = positive_examples + negative_examples
#     x_text = [clean_str(sent) for sent in x_text]
#     # Generate labels
#     positive_labels = [[0, 1] for _ in positive_examples]
#     negative_labels = [[1, 0] for _ in negative_examples]
#     y = np.concatenate([positive_labels, negative_labels], 0)
#     return [x_text, y]
#
#
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
        # 1 选择每次迭代,是否洗数据,像洗牌意义
        # 2 用生成器,每次只输出shuffled_data[start_index:end_index]这么多
    """
    data = np.array(data)
    print('the len of data is :', len(data))
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print(data_size, num_batches_per_epoch)  # 5 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # print(batch_num, shuffled_data[start_index:end_index])
            yield shuffled_data[start_index:end_index]


# new model
def load_data_and_labels(input_text_file, input_label_file, num_labels):
    # x_text = read_and_clean_zh_file(input_text_file)
    x_text = []
    with open(input_text_file, 'r', encoding='UTF-8') as file:
        for row in file:
            x_text.append(row)

    # y = None if not os.path.exists(input_label_file) else map(int, list(open(input_label_file, "r").readlines()))
    y = [row.strip('\n') for row in open(input_label_file, 'r')]
    return (x_text, y)


def load_positive_negative_data_files(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # positive_examples = read_and_clean_zh_file(positive_data_file)
    # negative_examples = read_and_clean_zh_file(negative_data_file)
    positive_examples = list(open(positive_data_file, "r", encoding='UTF-8'))
    # positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='UTF-8'))
    # negative_examples = [s.strip() for s in negative_examples]
    # Combine data
    # x_text = positive_examples + negative_examples
    # x_text = np.concatenate((positive_examples, negative_examples), 0)

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    i = 0
    x_ = []
    for line in positive_examples:
        line_ = list(line)
        line_.append(positive_labels[i])
        i += 1
        x_.append(line_)
    i = 0
    for line in negative_examples:
        line_ = list(line)
        line_.append(negative_labels[i])
        i += 1
        x_.append(line_)
    return x_  # [x_, y]


def padding_sentences(input_sentences, padding_token):
    # max_sentence_length = 5000 # 默认设定为5000
    # # 5000时候一行变量为50k大小，179334*50k = 8966700k = 8966M = 9G。。。
    max_sentence_length = 500
    # with open(input_sentences, 'r', encoding='UTF-8') as file:
    #     lines = file.readlines() # 速度快
    res = []
    for line in input_sentences:
        # import os
        # if os.path.exists(padding_file) == True:
        #     print('already exists! ')
        #     break # 如果存在则退出操作
        line = ((line[1]+line[2]).strip('\n').split(' ')[:500]) # 每行取title+str
        if len(line) > max_sentence_length:
            line = line[:max_sentence_length]
        else:
            line.extend([padding_token] * (max_sentence_length - len(line)))  # list格式
        # with open(padding_file, 'a', encoding='UTF-8') as file_save:
        #     file_save.write(' '.join(line) + '\n')
        res.append(line)
    return res


def batch_iter_my(x, y, batch_size, num_epochs, shuffle=False):
    '''
    Generate a batch iterator for a dataset
    '''
    data = y
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    print('num_batches_per_epoch :', num_batches_per_epoch)  # 2810
    print('data_size:', data_size)
    print('num_epochs:', num_epochs)
    # train = n个epoch
    # 1个epoch = n个batch
    # 1个batch = n个data
    for epoch in range(num_epochs):  # 可以考虑把输入的训练数据提前shuffle
        #     if shuffle:
        #     # Shuffle the data at each epoch
        #         shuffle_indices = np.random.permutation(np.arange(data_size))
        #         shuffled_data = data[shuffle_indices]
        #     else:
        #         shuffled_data = data
        print('epoch :', epoch)
        for batch_num in range(num_batches_per_epoch):
            # print('batch_num :', batch_num)
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            temp_x = []
            for i in range(end_idx - start_idx):
                # print('\t .. ', i)
                temp_x.append(next(x))  # 需要手动遍历迭代器
            # print(temp_x) # 每次都是最后特别慢。。

            # res = zip(temp_x, y[start_idx:end_idx]) # zip(*zip(x,y)) 解包x,y
            # res_ = np.array(list(res))
            # print('res:', res_)
            yield temp_x, y[start_idx:end_idx]
            # yield shuffled_data[start_idx : end_idx]


def batch_iter_for_test(x, length, batch_size, num_epochs, shuffle=False):
    '''
    Generate a batch iterator for a dataset
    '''
    data_size = length
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    print('num_batches_per_epoch :', num_batches_per_epoch)  # 2810
    # train = n个epoch
    # 1个epoch = n个batch
    # 1个batch = n个data
    for epoch in range(num_epochs):  # 可以考虑把输入的训练数据提前shuffle
        print('epoch :', epoch)
        for batch_num in range(num_batches_per_epoch):
            print('batch_num :', batch_num)
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            temp_x = []
            for i in range(end_idx - start_idx):
                # print('\t .. ', i)
                temp_x.append(next(x))  # 需要手动遍历迭代器
            # print(temp_x) # 每次都是最后特别慢。。
            yield temp_x


def mkdir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def seperate_line(line):
    return ''.join([word + ' ' for word in line])


def read_and_clean_zh_file(input_file, output_cleaned_file=None):
    lines = list(open(input_file, "r", encoding='UTF-8').readlines())
    lines = [clean_str(seperate_line(line)) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w', encoding='UTF-8') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))
    return lines


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(u"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()


def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:  # 采用二进制来读写，因为pickle默认为二进制形式
        pickle.dump(input_dict, f)


def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:  # 采用二进制来读写，因为pickle默认为二进制形式
        output_dict = pickle.load(f)
    return output_dict


'''
测试词向量word2vec api
    安装：
        pip install gensim
'''
from gensim.models import word2vec


def train_and_save_model(source_file, save_file):
    # source_file = './data/chinese_seg'
    # save_file = "./data/model/word2vec_test.model"

    # 1. train_and_test model
    sentences = word2vec.Text8Corpus(source_file)
    # sentences = [['第一', '我喜欢宝马'], ['第二', '我不喜欢奔驰']]
    model = Word2Vec(sentences, min_count=1, size=128)
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100


    # 2. save model
    # 保存模型，以便重用
    model.save(save_file)
    # 对应的加载方式
    # model_2 = word2vec.Word2Vec.load("text8.model")

    # 以一种C语言可以解析的形式存储词向量
    # model.save_word2vec_format(u"书评.model.bin", binary=True)
    # 对应的加载方式
    # model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)


def load_and_use_model(model_file):
    # model_file = "./data/model/word2vec_test.model"

    # 1. load model
    # 对应的加载方式
    model_ = Word2Vec.load(model_file)

    # 2. 导出权重和词库
    # 2.1 导出权重
    weight = model_.wv.syn0
    # print(weight)

    # 2.2 导出词库
    vocab = dict([(k, v.index) for k, v in model_.wv.vocab.items()])
    # print(vocab)

    return model_, weight, vocab

    # 3 将词输入到model输出weight
    # def word_to_id(word):
    #     id = vocab.get(word)
    #     if id is None:
    #         id = 0
    #     return id


    # # 4. use model
    # # 4.1 词之间相似度
    # y = model_.similarity('喜欢', '完美')
    # y2 = model_.similarity('喜欢', '吃饭')
    # print(y, y2)
    # # 4.2 一个词的最相似词
    # for i in model_.most_similar('吃饭'):
    #     print(i[0], i[1])


def merge_data(positive_data_file, negative_data_file, save_merge_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='UTF-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='UTF-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # print(x_text)
    # input()
    # x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    with open(save_merge_file, 'w', encoding='UTF-8') as file:
        for row in x_text:
            # print(row)
            # input()
            file.writelines(row + '\n')
            # return [x_text, y]


import jieba


def split_word(source_file, save_file):
    with open('./data/stopword.txt', 'r', encoding='UTF-8') as file:
        # s = {}.fromkeys(['w','1'])
        stopwords = {}.fromkeys([line.strip('\n') for line in file.readlines()])

        # stopwords = {}
        # for line in file.readlines():
        #     stopwords[line.strip('\n')] = None
    # 分词
    res = []
    with open(source_file, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            res.append(list(jieba.cut(line.strip('\n'), cut_all=False)))  # ' '.join()

    # 去停用词
    seg_filter_res = []
    for seg in res:
        filter = [word for word in seg if word not in stopwords and word != '']
        seg_filter_res.append(filter)

    # clean data
    # # pattern = re.compile(r'[^A-Za-z0-9]')
    # pattern = re.compile(r'[A-Za-z0-9]')
    # line = re.sub(pattern, '', line) # 去掉pattern词
    # print(line)

    # 保存结果
    file_save = open(save_file, 'w', encoding='UTF-8')
    for seg in seg_filter_res:
        # print(seg)
        file_save.writelines(' '.join(seg) + '\n')
    file_save.close()


def split_word_tsv(data_type, source_file, save_file):
    start_time_all = time.time()
    with open('./data/stopword.txt', 'r', encoding='UTF-8') as file:
        # s = {}.fromkeys(['w','1'])
        stopwords = {}.fromkeys([line.strip('\n') for line in file.readlines()])

    # 分词
    jieba.enable_parallel(4) # 不支持超线程，所以1*4核CPU
    res = []
    with open(source_file, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        count = 0
        len_lines = len(lines)
        for line in lines:
            count += 1
            start_time = time.time()
            str_list = line.strip('\n').split('\t')
            id_ = str_list[0]
            title_ = str_list[1]
            str_ = str_list[2]
            if data_type == 'train':
                label_ = str_list[3]
            elif data_type == 'test':
                pass
            else:
                print('please input data_type:')
                input()

            # jieba
            seg_title = list(jieba.cut(title_, cut_all=False))  # ' '.join()
            seg_str = list(jieba.cut(str_, cut_all=False))
            if data_type == 'train':
                result = id_ + '\t' + ' '.join(seg_title) + '\t' + ' '.join(seg_str) + '\t' + label_
            else:
                result = id_ + '\t' + ' '.join(seg_title) + '\t' + ' '.join(seg_str)

            res.append(result)
            stop_time = time.time()
            print('process:', count * 100 / len_lines, '%percent, cost:', stop_time - start_time)
            print('\t', data_type, ':', result)

    # # 去停用词
    # seg_filter_res = []
    # for seg in res:
    #     filter = [word for word in seg if word not in stopwords and word != '']
    #     seg_filter_res.append(filter)

    # clean data
    # # pattern = re.compile(r'[^A-Za-z0-9]')
    # pattern = re.compile(r'[A-Za-z0-9]')
    # line = re.sub(pattern, '', line) # 去掉pattern词
    # print(line)

    # 保存结果
    file_save = open(save_file, 'w', encoding='UTF-8')
    for line in res:
        print(line)
        file_save.writelines(line + '\n')
    file_save.close()

    stop_time_all = time.time()
    print('split word cost time：', stop_time_all - start_time_all)

class MySentences_tsv(object):
    # 自定义一次只读入一行的迭代器
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, 'r', encoding='UTF-8'):
            str_list = line.strip('\n').split('\t')
            id_ = str_list[0]
            title_ = str_list[1]
            str_ = str_list[2]
            # label_ = str_list[3] # 训练词向量用不到
            # clean data
            # ①大写字母转化为小写，同时过滤掉空格
            title_ = [item.lower() for item in title_.split(' ') if item != ' ']
            str_ = [item.lower() for item in str_.split(' ') if item != ' ']
            # merge to string
            title_ = ' '.join(title_)
            str_ = ' '.join(str_)
            # # ②只保留中文字符，并且保留分割效果
            # title_clean = ' '.join(re.findall(r'[\u4e00-\u9fffa-zA-Z]+', title_))
            # str_clean = ' '.join(re.findall(r'[\u4e00-\u9fffa-zA-Z]+', str_))
            res = id_ + ' ' + title_ + ' ' + str_
            print('res is:', res)
            yield res

class MySentences_clean_tsv(object):
    # 自定义一次只读入一行的迭代器
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        # c = 0
        for line in open(self.filename, 'r', encoding='UTF-8'):
            # c += 1
            # if c == 100:
            #     break
            str_list = line.strip('\n').split('\t')
            id_ = str_list[0]
            title_ = str_list[1]
            str_ = str_list[2]
            # label_ = str_list[3] # 训练词向量用不到
            # clean data
            # ①大写字母转化为小写，同时过滤掉空格
            title_ = [item.lower() for item in title_.split(' ') if item != ' ']
            str_ = [item.lower() for item in str_.split(' ') if item != ' ']
            # merge to string
            title_ = ' '.join(title_)
            str_ = ' '.join(str_)
            # ②只保留中文字符，并且保留分割效果
            title_clean = ' '.join(re.findall(r'[\u4e00-\u9FA5a-zA-Z，。,.]+', title_))
            str_clean = ' '.join(re.findall(r'[\u4e00-\u9FA5a-zA-Z，。,.]+', str_))
            res = (title_clean + ' ' + str_clean).split(' ')
            # 训练的输入是分好的句子，也可以是列表，也可以是得带起（class+iter），
            # 但不能是生成器（函数+yield），因为生成器只能遍历一次，而训练word2vec需要多次遍历数据
            print('res is:', res)
            yield res

# 训练词向量模型
def clean_and_trainWord2Vec_and_save_model(input_seg, output_model, clean=False): # 输入分好词的文件
    # 1. train_and_test model
    # sentences = word2vec.Text8Corpus('./data/chinese_seg')
    start_time = time.time()
    if clean:
        sentences = MySentences_clean_tsv(input_seg) # './data/BDCI2017-360/train_seg.tsv'
    else:
        sentences = MySentences_tsv(input_seg)
    model = Word2Vec(sentences, min_count=5, size=200, window=5, sg=1,
                     iter=20, workers=multiprocessing.cpu_count()) # sg=1使用skip算法
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    # 默认sg=0表示使用CBOW训练算法，使用skip-gram训练算法则设置sg=1,skip-gram训练慢，但是效果好


    # 2. save model
    # 保存模型，以便重用
    model.save(output_model) # "./data/model/word2vec_test.model"
    stop_time = time.time()
    print('word2vec cost time:', stop_time - start_time)

def load_and_split_test():
    save_file = './data/train_and_test/test_360_seg'
    split_word('./data/train_and_test/test_360', save_file)
    res = []
    with open(save_file, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            res.append(line)
    return res

def analyse_data():
    with open('./data/train_and_test/train_360_seg_x_text.bak', 'r',
              encoding='UTF-8') as file:  # train_360_seg_x_text.bak
        max_len = 0
        from matplotlib import pyplot as plt
        import pandas as pd
        data_res = []
        count = count1 = count5 = count6 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for row in file:
            now_len = len(row.split(' '))  # all       neg     pos
            if now_len > 20000:  # 12        5       7
                count1 += 1
            elif now_len > 10000:  # 37        27      10
                count2 += 1
            elif now_len > 6000:  # 156       89      67
                count3 += 1
            elif now_len > 5000:  # 163       94      69
                count4 += 1
            elif now_len > 1000:  # 20096
                count5 += 1
            elif now_len > 200:  # 134782
                count6 += 1
            else:  # 44552
                count += 1
                data_res.append(now_len)  # 1115164   35260   115164
            max_len = max(max_len, now_len)
    print('the count > 20000 is:', count)
    print('the count > 10000 is:', count2)
    print('the count > 6000 is:', count3)
    print('the count > 5000 is:', count4)
    print('the count > 1000 is:', count5)
    print('the count > 200 is:', count6)
    print('the left count is:', count)
    plt.scatter(x=range(len(data_res)), y=data_res, color='DarkBlue', label='analyse_data')
    plt.show()
    print('the max_len of line is', max_len)


def load_and_split_eval_test(source_file, save_file):
    # 加载真正提交结果的测试集

    with open('./data/stopword.txt', 'r', encoding='UTF-8') as file:
        stopwords = []
        for line in file.readlines():
            stopwords.append(line.strip('\n'))
    # print(stopwords)
    # 分词
    res = []
    with open(source_file, 'r', encoding='UTF-8') as file:
        for line in file:
            str_list = line.split('\t')
            id = str_list[0]
            topic = str_list[1]  # 后期再用
            sentence = str_list[2].strip('\n')
            res.append(list(jieba.cut(sentence, cut_all=False)))

    # 去停用词
    seg_filter_res = []
    for seg in res:
        filter = [word for word in seg if word not in stopwords and word != '']
        seg_filter_res.append(filter)

    # 保存结果
    file_save = open(save_file, 'w', encoding='UTF-8')
    for seg in seg_filter_res:
        # print(seg)
        file_save.writelines(' '.join(seg) + '\n')
    file_save.close()


# load and shuffle train data:
def load_train_data_file(train_file, clean=False):
    # 1. load train data
    with open(train_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        x_text = []
        count1 = 0
        for line in lines:
            # count1 += 1
            # if count1 <= 499901:
            #     continue
            str_list = line.strip('\n').split('\t')
            x_text.append(str_list)
        print(len(x_text))
    # 2. shuffle data
    # import random
    # random.seed(21)
    # random.shuffle(x_text)
    data_shuffled, y_shuffled = [], []
    count = count_neg = count_pos = 0
    for line in x_text:
        count += 1
        # if count >= 160001:
        #     break
        label_ = line[-1:][0] # 最后一个
        if label_ == 'NEGATIVE':
            y_temp = [1, 0]
            count_neg += 1
        elif label_ == 'POSITIVE':
            y_temp = [0, 1]
            count_pos += 1
        else:
            print(type(line[-1:]), line[-1:])
            input()
        # clean data：
        if clean:
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
        else:
            res = line[:-1]
            print(res)
        data_shuffled.append([res, y_temp])  # 取最后一个label数据
    print('the count is:', count, 'pos:', count_pos, 'neg:', count_neg)
    return data_shuffled


# 验证格式是否正确
def test_file():
    with open('../../BDCI2017-360/train_old_seg.tsv', 'r', encoding='utf-8') as file:
        start_time = time.time()
        print('starting...')
        lines = file.readlines()
        x_text = []
        for line in lines:
            str_list = line.strip('\n').split('\t')
            if len(str_list) == 4:
                id_ = str_list[0]
                title_ = str_list[1]
                str_ = str_list[2]
                label_ = str_list[3]
            else:
                print(str_list[0], ':', len(str_list))
            # x_text.append(str_list)
        stop_time = time.time()
        print('cost:', stop_time-start_time)

if __name__ == "__main__":
    # 1. 合并文件
    # merge_data('./data/rt-polaritydata/rt-polarity.pos', './data/rt-polaritydata/rt-polarity.neg')
    # merge_data('./data/train_and_test/train_pos', './data/train_and_test/train_neg', './data/train_and_test/train_360')

    # 2. 分词

    # load_and_split_eval_test('../../data/evaluation_public.tsv', './data/test_360_seg')
    # split_word_tsv('./data/BDCI2017-360/train.tsv', './data/BDCI2017-360/train_seg.tsv')
    # split_word_tsv('../../BDCI2017-360/train_old.tsv', '../../BDCI2017-360/train_old_seg.tsv')
    # split_word_tsv('../../BDCI2017-360/evaluation_public_old.tsv', '../../BDCI2017-360/evaluation_public_old_seg.tsv')

    # split_word_tsv('train', '../../BDCI2017-360/train.tsv', '../../BDCI2017-360/train_seg.tsv')



    # split_word_tsv('train', '../../BDCI2017-360/BDCI2017-360-Semi/train.tsv',
    #                '../../BDCI2017-360/BDCI2017-360-Semi/train_seg.tsv')
    # split_word_tsv('test', '../../BDCI2017-360/BDCI2017-360-Semi/evaluation_public.tsv',
    #                '../../BDCI2017-360/BDCI2017-360-Semi/evaluation_public_seg.tsv')

    clean_and_trainWord2Vec_and_save_model(input_seg='../../BDCI2017-360/BDCI2017-360-Semi/train_seg.tsv',
                         output_model='../../BDCI2017-360/model/word2vec_semi_final.model',
                                           clean=True)
    # word2vec cost time: 17566.843450784683




    pass

    # # 2. 训练Word2vec model
    # # train_and_save_model(source_file='./data/rt-polaritydata/rt', save_file='./data/rt-polaritydata/word2vec_rt.model')
    # train_and_save_model(source_file='./data/train_and_test/train_360_seg', save_file='./data/train_and_test/word2vec_360.model')

    # # 3. 加载Word2vec model
    # model, weight, vocab = load_and_use_model(model_file='./data/train_and_test/word2vec_360.model')
    # print(weight)
    # print(len(vocab))
    # print(vocab.get('北大'), vocab.get('。'))
    # print(model['北大'])
    # # print(model.train('阿姨 您好 这是 精心制作 宣传 资料'))

    # 4. 统计每行长度
    # analyse_data()

