"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""
import numpy as np
import pickle as pkl



from gensim.models import word2vec
def loadWord2Vec(filename):
    vocab, embd = [], []
    model_ = word2vec.Word2Vec.load(filename)

    # 2.1 导出权重
    weight = model_.wv.syn0
    # print(weight)
    print('weight:', len(weight))  # 506755 506633
    # 2.2 导出词库
    vocab = dict([(k, v.index) for k, v in model_.wv.vocab.items()])
    # print(vocab['外厅'])  # 返回的是词典，也就是k-v，对应: 词-序号，权重才是根据序号找到该词的embedding
    # print(len(vocab))  # 506755
    word_dim = int(len(weight[1]))


    vocab["UNKNOWN"] = len(vocab)   # vocab添加词的id，不能+1
    weight = weight.tolist()
    weight.append([0]*word_dim)  # embd添加相应的word2vec

    print("loaded word2vec, vocab:", len(vocab), 'weight:', len(weight))
    return vocab, weight

print("loading the pretrained word2vec...")
vocab, weight = loadWord2Vec(filename='../../BDCI2017-360/model/word2vec_semi_final_non.model')
vocab_size = len(vocab)
embedding_dim = len(weight[0])
embedding = np.asarray(weight)  # 当数据源是ndarray时，np.array()仍然会copy出一个副本，占用新的内存，而asarray不会
# print(weight[1])
# print(vocab[','], vocab['，'], vocab['.'], vocab['。'])
print(type(vocab))
# for i in vocab.keys():
#     if vocab[i] == 13:#vocab.values()
#         print(i)
# print(weight[0], weight[13])

# print(weight)


# input()
# firstsentence = '中国 经济 上半年 增长 强劲 ，'.split(' ')  # 保存句子中，每个词的ids
# firstsentence = np.array([vocab[item] for item in firstsentence])
# print(firstsentence.shape, firstsentence)




#file path
dataset_path='data/subj0.pkl'
dataset_360_10000_path = '../../BDCI2017-360/BDCI2017-360-Semi/train_seg_clean.pkl'
def set_dataset_path(path):
    dataset_path = path


def load_data(max_len, batch_size, n_words=20000, valid_portion=0.1, sort_by_len=True, usemydata=False):
    # 此为demo的原始数据
    if not usemydata:
        f = open(dataset_path, 'rb')
        print('load data from %s', dataset_path)
        train_set = np.array(pkl.load(f))
        test_set = np.array(pkl.load(f))
        f.close()

        train_set_x, train_set_y = train_set

        # train_set length
        n_samples = len(train_set_x)
        # shuffle and generate train and valid dataset
        sidx = np.random.permutation(n_samples)
        n_train = int(np.round(n_samples * (1. - valid_portion)))
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]

        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)

        # remove unknow words
        def remove_unk(x):
            return [[1 if w >= n_words else w for w in sen] for sen in x]

        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

        train_set_x = remove_unk(train_set_x)
        valid_set_x = remove_unk(valid_set_x)
        test_set_x = remove_unk(test_set_x)

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort_by_len:
            sorted_index = len_argsort(test_set_x)
            test_set_x = [test_set_x[i] for i in sorted_index]
            test_set_y = [test_set_y[i] for i in sorted_index]

            sorted_index = len_argsort(valid_set_x)
            valid_set_x = [valid_set_x[i] for i in sorted_index]
            valid_set_y = [valid_set_y[i] for i in sorted_index]

            sorted_index = len_argsort(train_set_x)
            train_set_x = [train_set_x[i] for i in sorted_index]
            train_set_y = [train_set_y[i] for i in sorted_index]

        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)
        test_set = (test_set_x, test_set_y)

        new_train_set_x = np.zeros([len(train_set[0]), max_len])
        new_train_set_y = np.zeros(len(train_set[0]))

        new_valid_set_x = np.zeros([len(valid_set[0]), max_len])
        new_valid_set_y = np.zeros(len(valid_set[0]))

        new_test_set_x = np.zeros([len(test_set[0]), max_len])
        new_test_set_y = np.zeros(len(test_set[0]))

        mask_train_x = np.zeros([max_len, len(train_set[0])])
        mask_test_x = np.zeros([max_len, len(test_set[0])])
        mask_valid_x = np.zeros([max_len, len(valid_set[0])])

        def padding_and_generate_mask(x, y, new_x, new_y, new_mask_x):

            for i, (x, y) in enumerate(zip(x, y)):
                # whether to remove sentences with length larger than maxlen
                if len(x) <= max_len:
                    new_x[i, 0:len(x)] = x
                    new_mask_x[0:len(x), i] = 1
                    new_y[i] = y
                else:
                    new_x[i] = (x[0:max_len])
                    new_mask_x[:, i] = 1
                    new_y[i] = y
            new_set = (new_x, new_y, new_mask_x)
            del new_x, new_y
            return new_set

        train_set = padding_and_generate_mask(train_set[0], train_set[1], new_train_set_x, new_train_set_y,
                                              mask_train_x)
        test_set = padding_and_generate_mask(test_set[0], test_set[1], new_test_set_x, new_test_set_y, mask_test_x)
        valid_set = padding_and_generate_mask(valid_set[0], valid_set[1], new_valid_set_x, new_valid_set_y,
                                              mask_valid_x)

        return train_set, valid_set, test_set

    else:
        # f = open(dataset_360_10000_path, 'rb')
        # print('load data from', dataset_360_10000_path)
        # test_set_raw = (pkl.load(f))
        # train_set_raw = (pkl.load(f))  # 说明是顺序存的两次数据
        # f.close()
        dataset_360_train = '../../BDCI2017-360/BDCI2017-360-Semi/train_seg.tsv'
        with open(dataset_360_train, 'r', encoding='utf-8') as file:
            count = 0
            test_set_raw, train_set_raw, valid_set_raw = [], [], []

            def isPos(x):
                if x == 'POSITIVE':
                    return 1
                else:
                    return 0
            for line in file.readlines():
                count += 1
                print('count:%6d, %.2f' %(count, count*100 / 600000), '%')
                if count >= 6001:
                    break
                str_list = line.strip('\n').split('\t')  # 单纯的jieba分词后的结果
                id_ = str_list[0]
                # temp = (str_list[1] + ' ' + str_list[2]).split(' ')[:max_len]  # 取前max_len长度的句子
                label_ = isPos(str_list[-1])
                if count >= 6000 - 60:  # 取后6000为test数据
                    test_set_raw.append([id_, label_])
                else:
                    train_set_raw.append([id_, label_])

        n_samples = len(train_set_raw)
        valid_set = train_set_raw[int(n_samples*(1. - valid_portion)):]
        train_set = train_set_raw[:int(n_samples*(1. - valid_portion))]

        print(len(train_set), len(valid_set), len(test_set_raw))


        # def padding_sentence(sentences):
        #     def padding(x):
        #         if len(x) <= max_len:
        #             for index in range(max_len - len(x)):
        #                 x.append('UNKNOWN')
        #         else:
        #             x = x[:max_len]
        #         return x
        #     tx, sequence_len, ty = [], [], []
        #
        #     for sentence in sentences:
        #         if len(sentence[0]) > max_len:
        #             save_len = max_len
        #         else:
        #             save_len = len(sentence[0])
        #         sequence_len.append(save_len)  # 每次进来之前先保存 原始句子的长度
        #         tx.append(padding(sentence[0]))  # 还是得先padding，然后只是在dynamic处理的时候只计算前面有效的部分。
        #         ty.append(sentence[1])
        #     return np.array(tx), np.array(sequence_len), np.array(ty)
        #
        #
        # train_set, valid_set, test_set = padding_sentence(train_set), padding_sentence(valid_set), padding_sentence(test_set_raw)


        # print('test memory:')
        # input()
        # del train_set_raw, test_set_raw
        # import gc
        # gc.collect()
        # print('finish clean memory')
        # input()


        # test_set_x, test_set_y = [], []
        # for i in test_set_raw[:100]:  # before 100
        #     tmp = []
        #     for j in i[0]:
        #         tmp.append(j)
        #     test_set_x.append(tmp)
        #     test_set_y.append(i[1])
        #
        # train_set_x, train_set_y = [], []
        # for i in train_set_raw[:1000]: # before 1000
        #     tmp = []
        #     for j in i[0]:
        #         # print(j)
        #         tmp.append(j)
        #     train_set_x.append(tmp)
        #     train_set_y.append(i[1])
        #
        # del train_set_raw, test_set_raw
        # import gc
        # gc.collect()
        #
        # print(np.array(test_set_x).shape, np.array(test_set_y).shape)
        # # sentences * sentence_length * word2vec
        # # list          list            np.array
        #
        # def isPos(x):
        #     if x == 'POSITIVE':
        #         return 1
        #     else:
        #         return 0
        # train_set_y = [isPos(i) for i in train_set_y]
        #
        # # train_set length
        # n_samples = len(train_set_x)
        # valid_set_x = train_set_x[int(n_samples*(1. - valid_portion)):]
        # valid_set_y = train_set_y[int(n_samples*(1. - valid_portion)):]
        # train_set_x = train_set_x[:int(n_samples*(1. - valid_portion))]
        # train_set_y = train_set_y[:int(n_samples*(1. - valid_portion))]
        #
        # test_set_y = [isPos(i) for i in test_set_y]
        #
        # # def len_argsort(seq): # 按长短排序，key为选择排序的元素，这里选择长度
        # #     return sorted(range(len(seq)), key=lambda x: len(seq[x]))
        # # # a = ['1', '123', '3213', '12']
        # # # 输出：[0, 3, 1, 2]
        # #
        # # if sort_by_len: # 排序后，就可以方便每个batch喂不同batch_size长度的数据了
        # #     sorted_index = len_argsort(test_set_x) # x，y要随同一个index变化
        # #     test_set_x = [test_set_x[i] for i in sorted_index]
        # #     test_set_y = [test_set_y[i] for i in sorted_index]
        # #
        # #     sorted_index = len_argsort(valid_set_x)
        # #     valid_set_x = [valid_set_x[i] for i in sorted_index]
        # #     valid_set_y = [valid_set_y[i] for i in sorted_index]
        # #
        # #     sorted_index = len_argsort(train_set_x)
        # #     train_set_x = [train_set_x[i] for i in sorted_index]
        # #     train_set_y = [train_set_y[i] for i in sorted_index]
        #
        # embeddingUnknow = [0 for i in range(200)]
        # def padding_sentence(sentences):
        #     def padding(x):
        #         # print(len(x), max_len)
        #         if len(x) <= max_len:
        #             for index in range(max_len - len(x)):
        #                 x.append(embeddingUnknow)
        #         else:
        #             x = x[:max_len]
        #         return x
        #     tx, sequence_len = [], []
        #
        #     for sentence in sentences:
        #         # print(len(sentence))
        #         if len(sentence) > max_len:
        #             save_len = max_len
        #         else:
        #             save_len = len(sentence)
        #         sequence_len.append(save_len) # 每次进来之前先保存 原始句子的长度
        #         tx.append(padding(sentence)) # 还是得先padding，然后只是在dynamic处理的时候只计算前面有效的部分。
        #         # tx.append(sentence)
        #     return np.array(tx), np.array(sequence_len)
        #
        # train_set = (padding_sentence(train_set_x), train_set_y) # 包括len
        # valid_set = (padding_sentence(valid_set_x), valid_set_y)
        # test_set = (padding_sentence(test_set_x), test_set_y)
        #
        # print('padding sentence shape:', train_set[0][0].shape)
        # print('padding sequence_len shape:', train_set[0][1].shape)
        # print('train_set_y:', train_set_y)
        # print('valid_set:', valid_set_y)
        # print('test_set:', test_set_y)

        return train_set, valid_set, test_set_raw

def read_n_line(start_index, end_index, max_len, clean=True):
    dataset_360_train = '../../BDCI2017-360/BDCI2017-360-Semi/train_seg.tsv'
    with open(dataset_360_train, 'r', encoding='utf-8') as file:
        count = 0
        sentence_len = []
        def padding_sentence(sentences):
            def padding(x):
                if len(x) <= max_len:
                    for index in range(max_len - len(x)):
                        x.append('UNKNOWN')
                else:
                    x = x[:max_len]
                return x
            tx, sequence_len, ty = [], [], []

            for sentence in sentences:
                if len(sentence[0]) > max_len:
                    save_len = max_len
                else:
                    save_len = len(sentence[0])
                sequence_len.append(save_len)  # 每次进来之前先保存 原始句子的长度
                tx.append(padding(sentence[0]))  # 还是得先padding，然后只是在dynamic处理的时候只计算前面有效的部分。
                ty.append(sentence[1])
            return np.array(tx), np.array(sequence_len), np.array(ty)

        def isPos(x):
            if x == 'POSITIVE':
                return 1
            else:
                return 0
        res = []
        for line in file.readlines()[start_index:end_index]:
            count += 1
            str_list = line.strip('\n').split('\t')  # 单纯的jieba分词后的结果
            temp = (str_list[1] + ' ' + str_list[2]).split(' ')[:max_len]  # 取前max_len长度的句子
            label_ = isPos(str_list[-1])
            # clean here
            if clean:
                pass

            res.append([temp, label_])
        # print('load file done!')
        # padding
        res = padding_sentence(res)  # tx, sequence_len, ty
        # print('padding sentence done!')
        return res

# return batch dataset
def batch_iter(max_len, data, batch_size, usemydata=False, pretrain_word2vec=True):
    if not usemydata:
        # get dataset and label
        x, y, mask_x = data
        x = np.array(x)
        y = np.array(y)
        data_size = len(x)
        num_batches_per_epoch = int((data_size - 1) / batch_size)
        for batch_index in range(num_batches_per_epoch):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, data_size)
            return_x = x[start_index:end_index]
            return_y = y[start_index:end_index]
            return_mask_x = mask_x[:, start_index:end_index]
            # if(len(return_x)<batch_size):
            #     print(len(return_x))
            #     print return_x
            #     print return_y
            #     print return_mask_x
            #     import sys
            #     sys.exit(0)
            yield (return_x, return_y, return_mask_x)
    else:
        if not pretrain_word2vec:
            # get dataset and label
            (x, sequence_len), y = data # ((train_set_x, sequence_len), train_set_y)
            x = np.array(x)
            y = np.array(y)
            sequence_len = np.array(sequence_len)
            # print('sequence_len:', sequence_len.shape, 'x:', x.shape, 'y:', y.shape)
            data_size=len(x)
            # print('len of data size:', x.shape, y.shape)

            num_batches_per_epoch = int((data_size-1) / batch_size) + 1
            for batch_index in range(num_batches_per_epoch):
                start_index = batch_index*batch_size
                end_index = min((batch_index+1)*batch_size,data_size)
                return_x = x[start_index:end_index]
                return_y = y[start_index:end_index]
                return_len = sequence_len[start_index:end_index]
                # return_mask_x = mask_x[:,start_index:end_index]

                yield (return_x, return_y, return_len) # ,return_mask_x)
        else:
            # get dataset and label
            data_size = len(data)

            def ids_to_embedding(sentences):  # 转换为ids表示
                global vocab
                res = []
                c = 0
                c+=1
                for sentence in sentences:
                    # if c == 1:
                    #     print('the len of sentence:', len(sentence))
                    #     print('the sentence:', sentence)
                    res_temp = []
                    for word in sentence:
                        if word in vocab:
                            res_temp.append(vocab[word])
                        else:
                            res_temp.append(vocab['UNKNOWN'])
                    # if c == 1:
                    #     print(res_temp)
                    res.append(res_temp)
                # print('ids_to_sentence done!')
                return np.array(res)

            num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
            print('num_batches_per_epoch:', num_batches_per_epoch)
            for batch_index in range(num_batches_per_epoch):
                print('batch_index:', batch_index)
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, data_size)
                return_x, return_len, return_y = read_n_line(start_index, end_index, max_len)
                return_x = ids_to_embedding(return_x)  # 转化为ids

                yield (return_x, return_y, return_len)  # ,return_mask_x)


