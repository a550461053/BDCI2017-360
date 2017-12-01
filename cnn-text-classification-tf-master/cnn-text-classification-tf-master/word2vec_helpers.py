# coding=utf-8

'''
测试词向量word2vec api
    安装：
        pip install gensim

'''
from gensim.models import word2vec # 我的model
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import sys
import logging
import multiprocessing
import time
import json
import numpy as np
import gc
import psutil

def output_vocab(vocab):
    for k, v in vocab.items():
        print(k)

class MySentences(object):
    # 自定义一次只读入一行的迭代器
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename, 'r', encoding='UTF-8'):
            yield line.split(' ')
class MySentences_tsv(object):
    # 自定义一次只读入一行的迭代器
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename, 'r', encoding='UTF-8'):
            str_ = line.strip('\n').split('\t')
            yield str_[1] + ' ' + str_[2]

def embedding_sentences(sentences_file, embedding_size=128, window=5, min_count=5, file_to_load=None, file_to_save=None):

    print('embedding_sentences...')
    if file_to_load is not None:
        print('\t load the file:', file_to_load)
        start_time = int(time.time())
        w2vModel = Word2Vec.load(file_to_load)
        stop_time = int(time.time())
        print('\t model loaded! cost time: ', stop_time-start_time)
        # print('\t load the sentences:', sentences_file)
        # sentences = MySentences(sentences_file)
        sentences = sentences_file
        # print('\t sentences loaded!')
    else:
        # 法一：直接读入内存后整体再给word2vec会占用大量ram，所以只能小文本的时候用
        # sentences = []
        # with open(sentences_file, 'r', encoding='UTF-8') as file:
        #     for line in file:
        #         temp = line.split(' ')[:5000]
        #         sentences.append(' '.join(temp)) # 只计算分词后，长度小于5000的部分
        # w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
        #                     workers=multiprocessing.cpu_count())

        # 法二：word2vec支持迭代器读取，可以自己定义一个迭代器，这样就可以一次读一条记录到ram中了
        # sentences = MySentences(sentences_file)
        sentences = sentences_file # input sentences lists
        start_time = int(time.time())
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
        stop_time = int(time.time())
        print('\t word2vec model has been trained! and cost time: ', stop_time-start_time)
        if file_to_save is not None:
            print('\t saving model file...')
            w2vModel.save(file_to_save)
            print('\t model saved!')
    # all_vectors = np.array([])
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    # start_time = int(time.time())
    # count = 0
    # for sentence in sentences:
    #     this_vector = np.array([])
    #     for word in sentence:
    #         if word in w2vModel.wv.vocab:
    #             this_vector = np.concatenate((this_vector, np.array(w2vModel[word])))
    #         else:
    #             this_vector = np.concatenate((this_vector, np.array(embeddingUnknown)))
    #     count += 1
    #     if count % 1000 == 0:
    #         last_memory = psutil.Process(os.getpid()).memory_info().rss
    #         gc.collect()
    #         now_memory = psutil.Process(os.getpid()).memory_info().rss
    #         print('now_memory is:', now_memory, 'reduce memory:', now_memory - last_memory)
    #     if count == 1:
    #         all_vectors = np.array([this_vector])
    #     else:
    #         all_vectors = np.concatenate((all_vectors, np.array([this_vector])))
    #     print(count, 'this_vectors size:', sys.getsizeof(this_vector), 'all_vectors size:', sys.getsizeof(all_vectors))
    # stop_time = int(time.time())
    # print('\t generate word2vec and cost time: ', stop_time - start_time)
    # print('the size of all_vectors is:', sys.getsizeof(all_vectors))
    # return all_vectors
    for sentence in sentences: #  in range(len(y)):
        # with open('./data/train_and_test/train_360_vector', 'a') as file:
        this_vector = []
        for word in sentence: # 手动遍历迭代器
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        # print('this vector is:', this_vector)
            # file.write(' '.join([str(item) for item in this_vector]) + '\n')
        yield this_vector


def generate_word2vec_files(input_file, output_model_file, output_vector_file, size=128, window=5, min_count=5):
    start_time = time.time()

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model = Word2Vec(LineSentence(input_file), size=size, window=window, min_count=min_count,
                     workers=multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))


def run_main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    input_file, output_model_file, output_vector_file = sys.argv[1:4]

    generate_word2vec_files(input_file, output_model_file, output_vector_file)


def test():
    vectors = embedding_sentences([['first', 'sentence'], ['second', 'sentence']], embedding_size=4, min_count=1)
    print(vectors)

# 训练词向量模型
def train_and_save_model(input_seg, output_model):
    # 1. train_and_test model
    # sentences = word2vec.Text8Corpus('./data/chinese_seg')
    sentences = MySentences_tsv(input_seg) # './data/BDCI2017-360/train_seg.tsv'
    model = word2vec.Word2Vec(sentences, min_count=3, size=200, window=5)
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100

    # 2. save model
    # 保存模型，以便重用
    model.save(output_model) # "./data/model/word2vec_test.model"
    # 对应的加载方式
    # model_2 = word2vec.Word2Vec.load("text8.model")

    # 以一种C语言可以解析的形式存储词向量
    # model.save_word2vec_format(u"书评.model.bin", binary=True)
    # 对应的加载方式
    # model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)


def load_and_use_model(model_file):
    # 1. load model
    # 对应的加载方式
    model_ = Word2Vec.load(model_file)
    print(model_)
    # 2. 导出权重和词库
    # 2.1 导出权重
    weight = model_.wv.syn0
    print(weight)

    # 2.2 导出词库
    vocab = dict([(k, v.index) for k, v in model_.wv.vocab.items()])
    print(vocab)
    print(len(vocab))
    for word in ['房间', '宝马']:
        for i in model_.most_similar(word):
            print(i[0], i[1])

    # 3. use model
    # 3.1 词之间相似度
    y = model_.similarity('喜欢', '完美')
    y2 = model_.similarity('喜欢', '吃饭')
    print(y, y2)
    # 3.2 一个词的最相似词
    for i in model_.most_similar('吃饭'):
        print(i[0], i[1])

# save 360 分词后的词向量矩阵 data to pickle
def word2vec_and_save_data(model_file, file_path, file_save_path, percent=0.01):

    w2vModel = Word2Vec.load(model_file)
    print(w2vModel) # 494796
    # 2. 导出权重和词库
    # 2.1 导出权重
    # weight = w2vModel.wv.syn0
    # print(weight)

    embeddingDim = w2vModel.vector_size
    embeddingUnknow = [0 for i in range(embeddingDim)]

    # 2.2 导出词库
    # vocab = dict([(k, v.index) for k, v in w2vModel.wv.vocab.items()])

    # 3. 保存数据：[str,label]
    # train_set = test_set = [] # 不能这样写，，，
    train_set, test_set = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data_size = len(lines)
        train_test_size = data_size * (1 - percent)
        print('data_size:', data_size, 'test_size:', train_test_size)
        count = 0
        for line in lines:
            count += 1
            if count > 10000:
                break
            x_vector = []
            str_list = line.strip('\n').split('\t')
            words = (str_list[1] + str_list[2]).strip('\n').split(' ')
            for word in words:
                if word in w2vModel.wv.vocab:
                    x_vector.append(w2vModel[word])
                else:
                    x_vector.append(embeddingUnknow)

            if count > 9000:# >= train_test_size:
                test_set.append([x_vector, str_list[-1]])
            else:
                train_set.append([x_vector, str_list[-1]])
            print('count:', count, 'x_vector len:', len(x_vector))
    print('the train_set:', len(train_set), 'the test_set:', len(test_set))
    import pickle
    fw = open(file_save_path, 'wb')
    print('now memory:', psutil.Process(os.getpid()).memory_info().rss)
    pickle.dump(test_set, fw, -1)
    import gc
    del test_set
    gc.collect()
    print('save test done! memory:', psutil.Process(os.getpid()).memory_info().rss)

    pickle.dump(train_set, fw, -1)
    fw.close()
    print('save all done!')


if __name__ == "__main__":
    # train_and_save_model()
    pass

    word2vec_and_save_data(model_file='../../BDCI2017-360/model/word2vec_semi_final.model',
                           file_path='../../BDCI2017-360/BDCI2017-360-Semi/train_seg_clean.tsv',
                           file_save_path='../../BDCI2017-360/BDCI2017-360-Semi/train_seg_clean.pkl', percent=0.01)


    # load_and_use_model(model_file='../../BDCI2017-360/model/word2vec_test.model')
    # train_and_save_model(input_seg='../../BDCI2017-360/train_old_seg.tsv',
    #                      output_model='../../BDCI2017-360/model/word2vec_test.model')

