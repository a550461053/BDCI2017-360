# coding=utf-8

'''
简单预处理
'''
import jieba
import re
import pandas as pd


# 分割为pos和neg两个文件： 从train数据的后100pos和后100neg作为训练集
def save_pos_neg():
    pos_ = []
    neg_ = []
    with open('./cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/train_and_test/train_360', 'r', encoding='UTF-8') as file: # data/train.tsv
        lines = file.readlines()
        count = 0
        for line in lines:
            str_list = line.split('\t')
            # print(len(str_list), str_list[0],str_list[1],str_list[3])
            count += 1
            if str_list[3] == 'POSITIVE' or str_list[3][0:8] == 'POSITIVE' or str_list[3][1:9] == 'POSITIVE':
                path = 'data/train_pos'
                pos_.append([str_list[0], str_list[1], str_list[2]])
            elif str_list[3] == 'NEGATIVE' or str_list[3][0:8] == 'NEGATIVE' or str_list[3][1:9] == 'NEGATIVE':
                path = 'data/train_neg'
                neg_.append([str_list[0], str_list[1], str_list[2]])
            # input()
    print(count, len(pos_), len(neg_)) # 200000 120163 79837
    count = 0
    with open('./cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/train_and_test/train_360_pos', 'w', encoding='UTF-8') as file: # data/train_pos
        for row in pos_:
            count += 1
            if count <= 101:
                # count = 0
                continue
            # file.writelines([row[0],'\t',row[1],'\t', row[2], '\n'])
            file.writelines([row[2], '\n'])
    with open('./cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/train_and_test/train_360_neg', 'w', encoding='UTF-8') as file: # data/train_neg
        count = 0
        for row in neg_:
            count += 1
            if count <= 101:
                continue
            # file.writelines([row[0],'\t',row[1],'\t', row[2], '\n'])
            file.writelines([row[2], '\n'])
    print('finish!')

def analyse_data():
    with open('./BDCI2017-360/train.tsv', 'r', encoding='utf-8') as file:
        count = 0
        lines = file.readlines()
        print(len(lines))
        for line in lines:
            count += 1
            str_list = line.strip('\n').split('\t')
            if len(str_list) != 4:
                print(len(str_list), str_list[0], str_list[-1:])
            else:
                id_ = str_list[0]
                title_ = str_list[1]
                str_ = str_list[2]
                label_ = str_list[3]
                pattern = re.compile(r'[\u4e00-\u9fa5]') # 或者：^[\u4E00-\u9FFF]+$
                temp = re.sub(pattern, '', str_)
                pattern2 = re.compile(r'[a-zA-Z0-9]')
                temp = re.sub(pattern2, '', temp)
                punctuation = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）《》；;-:°±℃{}]+"
                temp = re.sub(punctuation, '', temp)

                print(temp)
                # pass
        print('count is:', count)

        # evaluation_public.tsv 400000
        # train.tsv             500000

# 区分两次数据的异同
def find_diff():
    # train:
    # with open('./data/train.tsv', 'r', encoding='utf-8') as old_file:
    #     old_id = []
    #     for line in old_file.readlines():
    #         str_list = line.strip('\n').split('\t')
    #         id_ = str_list[0]
    #         old_id.append(id_)
    # with open('./BDCI2017-360/train.tsv', 'r', encoding='utf-8') as new_file:
    #     new_line = []
    #     count = 0
    #     for line in new_file.readlines():
    #         str_list = line.strip('\n').split('\t')
    #         id_ = str_list[0]
    #         if id_ not in old_id:
    #             count += 1
    #             new_line.append(str_list)
    # print('the new count of train:', count)
    # with open('./BDCI2017-360/train_change', 'w', encoding='utf-8') as file:
    #     for line in new_line:
    #         file.writelines('\t'.join(line) + '\n')
    # print('save train change!')
    # del old_id, new_line
    # import gc
    # gc.collect()

    # test:
    with open('./data/evaluation_public.tsv', 'r', encoding='utf-8') as old_file:
        old_id = []
        for line in old_file.readlines():
            str_list = line.strip('\n').split('\t')
            id_ = str_list[0]
            old_id.append(id_)
    with open('./BDCI2017-360/evaluation_public.tsv', 'r', encoding='utf-8') as new_file:
        new_line = []
        count = 0
        for line in new_file.readlines():
            str_list = line.strip('\n').split('\t')
            id_ = str_list[0]
            if id_ not in old_id:
                count += 1
                new_line.append(str_list)
    print('the new count of test:', count)
    with open('./BDCI2017-360/evaluation_public_change', 'w', encoding='utf-8') as file:
        for line in new_line:
            file.writelines('\t'.join(line) + '\n')
    print('save test change!')



#####################################

fast_path = './result/'
def pro_result_for_cnn(input_file, output_file):

    with open('./BDCI2017-360/BDCI2017-360-Semi/evaluation_public_seg.tsv', 'r', encoding='utf-8') as old_file:
        res_list = []
        for line in old_file.readlines():
            str_list = line.strip('\n').split('\t')
            res_list.append(str_list[0])
    with open(input_file, 'r', encoding='utf-8') as file:
        res_label = []
        count_pos = 0
        for line in file.readlines():
            if line.strip() == '0.0':
                temp = 'NEGATIVE'
                count_pos += 1
            elif line.strip() == '1.0':
                temp = 'POSITIVE'
            else:
                print('error:', line)
            res_label.append(temp)
    with open(output_file, 'w', encoding='utf-8') as save_file:
        i = 0
        for line in res_list:
            print(i)
            save_file.writelines(line + ',' + res_label[i] + '\n')
            i += 1
    print('the pos count:', count_pos)
    print('the count:', i)
    print('pos / count:', count_pos / i)
pro_result_for_cnn(input_file='result',
                        output_file='semi_test_cnn_res.csv')
input()

#########################################################################
#  for fasttext                                                         #
#########################################################################
def clean_data_for_fasttext():
    fast_path = '/home/batista/project/project/pycharm-python35/fastText/test360/'
    with open('./BDCI2017-360/BDCI2017-360-Semi/train_seg.tsv', 'r', encoding='utf-8') as train_seg:
        lines = train_seg.readlines() # evaluation_public
        res = []
        for line in lines:
            str_list = line.strip('\n').split('\t')
            id_ = str_list[0]
            title_ = str_list[1]
            str_ = str_list[2]
            label_ = str_list[3]
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
            res.append(id_ + '\t' + title_clean + '\t' + str_clean + '\t' + label_)
            # res.append(id_ + '\t' + title_clean + '\t' + str_clean)
    print('finish clean!')
    with open('./BDCI2017-360/BDCI2017-360-Semi/train_seg_clean.tsv', 'w', encoding='utf-8') as train_seg_clean:
        for line in res:
            train_seg_clean.write(line + '\n')

    print('finish all.')
# clean_data_for_fasttext()
# input()

def pro_data_for_fasttext():
    fast_path = '/home/batista/project/project/pycharm-python35/fastText/test360/'
    # train:
    with open('./BDCI2017-360/BDCI2017-360-Semi/train_seg_clean.tsv', 'r', encoding='utf-8') as old_file:
        res_list = []
        for line in old_file.readlines():
            str_list = line.strip('\n').split('\t')
            label = 1
            print(str_list[-1:], str_list[3])
            if str_list[3] == 'POSITIVE':
                label = '__label__POSITIVE'
            elif str_list[3] == 'NEGATIVE':
                label = '__label__NEGATIVE'
            else:
                print('error', str_list)
                input()
            res_list.append(label + '\t'+(str_list[1]+str_list[2]))
    # save_train:
    with open(fast_path+'train_semi_final.txt', 'w', encoding='utf-8') as save_file:
        for line in res_list:
            save_file.writelines(line + '\n')
    print('train_finished!')

    # test：
    with open('./BDCI2017-360/BDCI2017-360-Semi/evaluation_public_seg_clean.tsv', 'r', encoding='utf-8') as old_file:
        res_list = []
        for line in old_file.readlines():
            str_list = line.strip('\n').split('\t')
            print(str_list[-1:], str_list[2])
            res_list.append(str_list[1]+str_list[2])
    # save_test:
    with open(fast_path+'test_semi_final.txt', 'w', encoding='utf-8') as save_file:
        for line in res_list:
            save_file.writelines(line + '\n')
    print('test_finished!')
# pro_data_for_fasttext()
# input()

# 检查change和old是否对齐all
# with open('./BDCI2017-360/evaluation_public_change.tsv', 'r', encoding='utf-8') as old_file:
#     i = 0
#     change = {}
#     for line in old_file.readlines():
#         str_list = line.strip('\n').split('\t')
#         i += 1
#         change[str_list[1] + ' ' + str_list[2]] = None
#     print(i)
# with open('./BDCI2017-360/evaluation_public_old.tsv', 'r', encoding='utf-8') as old_file:
#     i = 0
#     old = {}
#     for line in old_file.readlines():
#         str_list = line.strip('\n').split('\t')
#         old[str_list[1] + ' ' + str_list[2]] = None
#         if line in change:
#             print(line)
#         i += 1
#     print(i)
# with open('./BDCI2017-360/evaluation_public.tsv', 'r', encoding='utf-8') as old_file:
#     i = 0
#     all = {}
#     for line in old_file.readlines():
#         str_list = line.strip('\n').split('\t')
#         i += 1
#         all[str_list[1] + ' ' + str_list[2]] = None
#     print(i)
# c = 0
#
# for i in old: # 之前的数据有77个不存在了
#     if i not in all:
#         print(i)
#         c += 1
# print(c)
# print(len(change), len(old), len(all))
# input()

fast_path = '/home/batista/project/project/pycharm-python35/fastText/test360/result/'
def pro_result_for_fasttext(input_file, output_file):

    with open('./BDCI2017-360/BDCI2017-360-Semi/evaluation_public_seg.tsv', 'r', encoding='utf-8') as old_file:
        res_list = []
        for line in old_file.readlines():
            str_list = line.strip('\n').split('\t')
            res_list.append(str_list[0])
    with open(input_file, 'r', encoding='utf-8') as file:
        res_label = []
        count_pos = 0
        for line in file.readlines():
            if line.strip() == '__label__NEGATIVE':
                temp = 'NEGATIVE'
                count_pos += 1
            elif line.strip() == '__label__POSITIVE':
                temp = 'POSITIVE'
            else:
                print('error:', line)
            res_label.append(temp)
    with open(output_file, 'w', encoding='utf-8') as save_file:
        i = 0
        for line in res_list:
            print(i)
            save_file.writelines(line + ',' + res_label[i] + '\n')
            i += 1
    print('the pos count:', count_pos)
    print('the count:', i)
    print('pos / count:', count_pos / i)
pro_result_for_fasttext(input_file=fast_path + 'semi_test_predict1.csv',
                        output_file=fast_path + 'semi_test_predict1_res.csv')
input()
# 单纯fasttext结果：
# the count: 400000
# pos / count: 0.6165475


# lr ws epoch wordNgrams loss     - 正负比例<0.8      score
# 5 50 0.590            -  0.811
# 8 50 0.584            -  0.811
# 9 50 0.580
# 12 50 0.569           - 0.8118
# 0.01 12 100 0.491     - 0.67 - 0.4 过拟合
# 12 60 0.5             - 0.717 - 0.37
# 0.1 12 55 0.5068      -
# 0.005 12 55 0.615     - 0.86
# 0.005 12 100 0.555    - 0.795 - 0.32
# 0.05 12 100 0.438     - 0.662 -
# 1.0 12 100 0.434      - 0.668 -
# 1.0 12 100 5 0.06     - 0.683 - 0.43
# 1.0 12 100 5 0.056 200 5000000 - 0.708 - 0.426
# 1.0 12 120 5 0.

def test_result():
    print('read data test:')

    data = pd.read_csv('./result/baseline_test_new.csv')

    # with open('./BDCI2017-360/evaluation_public.tsv', 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         str_list = line.strip().split('\t')
    #         print(str_list[:-1])
    #         # ['265bf687c9d3a51e0da90a6d37ee6ad2', '老百姓的白衣天使']
    #         # ['943948c68fb14f5c8dc8b37396c10587', '外媒称小米获10亿美元再融资 雷军微博点赞疑似默认']
    # print('读入数据：', data)

    count_neg = 0
    # print(data.loc[1])
    # print(data.index, data.columns)
    # data.describe()
    for i in data.loc[:,'POSITIVE']:
        print(i)
        if i == 'NEGATIVE':
            count_neg += 1
    print(count_neg)
    print(count_neg * 100.0 / 399999)
    #
    # 121343, 比第一次数据提交结果多了7w负样本，acc=0.55867
    # 86950，改为0.9的随机率，比第一次数据提交结果多了3w负样本，acc=0.58134
    # 69210，改为0.95的随机率，比第一次数据提交结果多了2w负样本，acc=逻辑错误
    # 79817，改为0.92的随机率，比第一次数据提交结果多了3w负样本，acc=逻辑错误
    # 86799，还是改为0.9靠谱，acc=0.5831
    # 3次同样的结果，每次相差0.001
    # for i in data:


## feature
# 设计特征工程
def select_1():

    with open('./BDCI2017-360/train.tsv', 'r', encoding='utf-8') as file:
        pass
        lines = file.readlines()
        count_same = i = c1 = c2 = c3 = 0 # 含有：用微信扫描二维码分享至好友和朋友圈
        str_1 = '用微信扫描二维码分享至好友和朋友圈'
        for line in lines:
            line_list = line.strip().split('\t')
            id_ = line_list[0]
            title_ = line_list[1]
            content_ = line_list[2]
            label_ = line_list[3]
            # m1 = re.findall(str_1, title_ + content_)
            m1 = re.findall(str_1, content_) # 2次连续
            if len(m1) >= 3: # 1为13237，>1为13548
                c1 += 1

                if label_ == 'POSITIVE': # NEGATIVE
                    i += 1
                    # print('id', id_, 'label_', label_, content_)
                    pass
                else:
                    # print('id', id_, 'label_', label_, content_)
                    pass
        print('the count same is:', c1, 'i is:', i)
        # count_same: 24357
        # 其中是POS的有：8976个
        # POS：含有重复1次：20，重复2次：8956，重复两次都是在content开头，重复3次：0
        # NEG：含有重复1次：53，重复2次：13421，重复3次：1907，
        #........................................
# select_1()

def select_2(): # 筛选出5w的负样本
    with open('./BDCI2017-360/train.tsv', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        count_error = count_pos = count_neg = 0
        flag = 'POSITIVE'
        for line in lines:
            line_list = line.strip().split('\t')
            id_ = line_list[0]
            title_ = line_list[1]
            s0 = content = line_list[2]
            label_ = line_list[3]

            if ("，的" in s0) and (not("，的确" in s0)) and (not("，的士" in s0)) \
                                and (not("，的哥" in s0)) and (not("，的的确确" in s0)):
                flag = "NEGATIVE"
            if ("，了" in s0) and (not("，了解" in s0)) and (not("，了结" in s0)) \
                                and (not("，了无" in s0)) and (not("，了却" in s0)) \
                                and (not("，了不起" in s0)):
                flag = "NEGATIVE"

            if ("。的" in s0) and (not("。的确" in s0)) and (not("。的士" in s0)) \
                                and (not("。的哥" in s0)) and (not("。的的确确" in s0)):
                flag = "NEGATIVE"
            if ("。了" in s0) and (not("。了解" in s0)) and (not("。了结" in s0)) \
                                and (not("。了无" in s0)) and (not("。了却" in s0)) \
                                and (not("。了不起" in s0)):
                flag = "NEGATIVE"

            if (";的" in s0) and (not(";的确" in s0)) and (not(";的士" in s0)) \
                                and (not(";的哥" in s0)) and (not(";的的确确" in s0)):
                flag = "NEGATIVE"
            if (";了" in s0) and (not(";了解" in s0)) and (not(";了结" in s0)) \
                                and (not(";了无" in s0)) and (not(";了却" in s0)) \
                                and (not(";了不起" in s0)):
                flag = "NEGATIVE"

            if ("?的" in s0) and (not("?的确" in s0)) and (not("?的士" in s0)) \
                                and (not("?的哥" in s0)) and (not("?的的确确" in s0)):
                flag = "NEGATIVE"
            if ("?了" in s0) and (not("?了解" in s0)) and (not("?了结" in s0)) \
                                and (not("?了无" in s0)) and (not("?了却" in s0)) \
                                and (not("?了不起" in s0)):
                flag = "NEGATIVE"

            if flag != label_:
                count_error += 1
            if label_ == 'POSITIVE':
                count_pos += 1
            if label_ == 'NEGATIVE':
                count_neg += 1
            flag = 'POSITIVE'
    print('count pos, neg, error:', count_pos, count_neg, count_error)

# clean train data:
def find_none_data():
    with open('./BDCI2017-360/train.tsv', 'r', encoding='utf-8') as file:
        count = count_title_none = count_content_none = c_N = chong = chong_pos = 0
        dict_ = {}
        for line in file.readlines():
            count += 1
            str_list = line.strip('\n').split('\t')
            id_, title_, content_, label_ = str_list[0], str_list[1], str_list[2],str_list[3]
            if title_ not in dict_:
                dict_[content_] = None
            else:
                chong += 1
                if label_ == 'POSITIVE':
                    chong_pos += 1
                print(label_, title_)
            if len(content_) >= 0 and len(content_) < 167: # 所有字数目,167是453个，只有一个是pos其他都是neg
                count_content_none += 1
                if label_ == 'NEGATIVE':
                    c_N += 1
        print('chong fu:', chong, chong_pos) # 重复的title：28113 7848
        # 重复的content_:
        print(count, count_title_none, c_N, count_content_none)

# find_none_data()

# 单个句子长度统计
def count_one_sentence():

    with open('./BDCI2017-360/train.tsv', 'r', encoding='utf-8') as file:
        count = count_temp_pos = count_temp_neg = count_content_none = c_N = toolang_pos = toolang_neg = 0
        dict_ = {}
        for line in file.readlines():
            count += 1
            str_list = line.strip('\n').split('\t')
            id_, title_, content_, label_ = str_list[0], str_list[1], str_list[2],str_list[3]
            content_list = content_.replace(' ','').replace('，', '。').replace('、', '。').replace('.', '。').replace('!', '。').replace('！', '。').replace(':', '。').replace('：', '。').replace('；', '。').split('。')
            for i in content_list:
                if len(i) > 100 and label_ == 'POSITIVE':
                    count_temp_pos += 1
                elif len(i) > 100 and label_ == 'NEGATIVE':
                    count_temp_neg += 1
            if count_temp_neg > 2:
                toolang_neg += 1
            elif count_temp_pos > 2:
                toolang_pos += 1
            count_temp_neg = count_temp_pos = 0
        # 重复的content_:
        print(count, toolang_pos, toolang_neg)
count_one_sentence()

pass