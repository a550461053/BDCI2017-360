# coding=utf-8
import pandas as pd
import time
import jieba
import jieba.posseg
import os, sys
import re


def go_split(s, min_len):
    # 拼接正则表达式
    symbol = '，；。！、？!'
    symbol = "[" + symbol + "]+"
    # 一次性分割字符串
    result = re.split(symbol, s)

    return [x for x in result if len(x) > min_len]


def is_dup(s, min_len):
    result = go_split(s, min_len)
    return len(result) != len(set(result))


def is_neg_symbol(uchar):
    neg_symbol = ['!', '0', ';', '?', '、', '。', '，']
    return uchar in neg_symbol


# #trick1
def process1():
    # 1. load data
    print('load data...')
    labelfile = './result/baseline_test.csv'
    other_label = pd.read_csv(labelfile, names=['id', 'label'])
    trainfile = './BDCI2017-360/evaluation_public_old.tsv'
    data = pd.read_csv(trainfile, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])
    print('the len of data:', len(data))
    data['label_pre'] = other_label.label


    # 2. eval data
    print('eval data...')
    data['content_is_dup'] = data['content'].apply(lambda x: is_dup(str(x), 10))
    data['label_pre'][data['content_is_dup']] = 'NEGATIVE'

    data['last_str'] = data['content'].apply(lambda x: str(x)[-1])
    data['last_str_is_neg_symbol'] = data['last_str'].apply(lambda x: is_neg_symbol(x))
    data['label_pre'][~data['last_str_is_neg_symbol']] = 'POSITIVE'

    data[['id', 'label_pre']].to_csv('result/trick1.csv', header=False, index=False)
# process1()
# print('finish!')
# input()

# 只是填充没有筛选的label 3. process
def is_nan(x):
    return x != 'NEGATIVE'

def process2():
    print('process data...')
    other_res = pd.read_csv('result/trick1.csv', names=['id', 'label_pre'])
    other_res['label'] = other_res.label_pre
    other_res['label_is_nan'] = other_res['label_pre'].apply(lambda x: is_nan(x)) # 返回0 or 1
    other_res['label'][other_res['label_is_nan']] = 'POSITIVE' # 按照是否0-1，分别赋值
    print(other_res[['id', 'label_pre']])
    print(other_res[['id', 'label']])

    other_res[['id', 'label']].to_csv('result/trick1_my.csv', header=False, index=False)

#1: 0.58134 -> 0.60284


def is_neg1(x):
    s0 = x
    flag = "POSITIVE"
    if ("，的" in s0) and (not ("，的确" in s0)) and (not ("，的士" in s0)) \
            and (not ("，的哥" in s0)) and (not ("，的的确确" in s0)):
        flag = "NEGATIVE"
    if ("，了" in s0) and (not ("，了解" in s0)) and (not ("，了结" in s0)) \
            and (not ("，了无" in s0)) and (not ("，了却" in s0)) \
            and (not ("，了不起" in s0)):
        flag = "NEGATIVE"

    if ("。的" in s0) and (not ("。的确" in s0)) and (not ("。的士" in s0)) \
            and (not ("。的哥" in s0)) and (not ("。的的确确" in s0)):
        flag = "NEGATIVE"
    if ("。了" in s0) and (not ("。了解" in s0)) and (not ("。了结" in s0)) \
            and (not ("。了无" in s0)) and (not ("。了却" in s0)) \
            and (not ("。了不起" in s0)):
        flag = "NEGATIVE"

    if (";的" in s0) and (not (";的确" in s0)) and (not (";的士" in s0)) \
            and (not (";的哥" in s0)) and (not (";的的确确" in s0)):
        flag = "NEGATIVE"
    if (";了" in s0) and (not (";了解" in s0)) and (not (";了结" in s0)) \
            and (not (";了无" in s0)) and (not (";了却" in s0)) \
            and (not (";了不起" in s0)):
        flag = "NEGATIVE"

    if ("?的" in s0) and (not ("?的确" in s0)) and (not ("?的士" in s0)) \
            and (not ("?的哥" in s0)) and (not ("?的的确确" in s0)):
        flag = "NEGATIVE"
    if ("?了" in s0) and (not ("?了解" in s0)) and (not ("?了结" in s0)) \
            and (not ("?了无" in s0)) and (not ("?了却" in s0)) \
            and (not ("?了不起" in s0)):
        flag = "NEGATIVE"
    if flag == "NEGATIVE":
        return 1
    else:
        return 0

def process3():
    print('process data...')
    testfile = './BDCI2017-360/evaluation_public.tsv'
    data = pd.read_csv(testfile, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])
    other_res = pd.read_csv('result/trick1_my.csv', names=['id', 'label'])
    data['label3'] = other_res.label
    data['content_is_neg1'] = data['content'].apply(lambda x: is_neg1(x))  # 返回0 or 1
    data['label3'][data['content_is_neg1']] = 'NEGATIVE'  # 按照是否0-1，分别赋值

    data[['id', 'label3']].to_csv('result/trick3_my.csv', header=False, index=False)
#2: 0.60284 -> 0.60252


def is_neg2(x):
    str_1 = '用微信扫描二维码分享至好友和朋友圈'
    m1 = re.findall(str_1, x)  # 2次连续

    return len(m1) >= 3  # 1为13237，>1为13548
def process4():
    print('process data...')
    testfile = './BDCI2017-360/evaluation_public.tsv'
    data = pd.read_csv(testfile, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])
    other_res = pd.read_csv('result/trick3_my.csv', names=['id', 'label'])
    data['label4'] = other_res.label
    data['content_is_neg1'] = data['content'].apply(lambda x: is_neg2(x))  # 返回0 or 1
    data['label4'][data['content_is_neg1']] = 'NEGATIVE'  # 按照是否0-1，分别赋值

    data[['id', 'label4']].to_csv('result/trick4_my.csv', header=False, index=False)
# 3: 0.60252 -> 0.60378


# 处理fasttext结果：fast_predict_0.1_300_5_50_hs.csv
def process_fasttext():
    print('process data...')
    testfile = './BDCI2017-360/evaluation_public_change.tsv'
    raw_data = pd.read_csv(testfile, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])

    fasttext = '/home/batista/project/project/pycharm-python35/fastText/test360/result/test_predict_res.csv'
    data = pd.read_csv(fasttext, sep=',', header=None, encoding='utf-8', names=['id', 'label'])
    data['content_is_neg1'] = raw_data['content'].apply(lambda x: is_neg2(x))  # 返回0 or 1
    data['label'][data['content_is_neg1']] = 'NEGATIVE'  # 按照是否0-1，分别赋值

    data[['id', 'label']].to_csv('/home/batista/project/project/pycharm-python35/fastText/test360/result/fasttext_trick5_change.csv', header=False, index=False)

# process_fasttext()
# process4()

def process_data_for_fasttext():
    print('process data for fasttext...')

def merge_change_and_old_data():
    print('merging...')
    old_file = 'result/trick1.csv'
    # raw_data = pd.read_csv(old_file, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])
    all_file = './BDCI2017-360/evaluation_public.tsv'
    # all_data = pd.read_csv(all_file, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])
    change_file = '/home/batista/project/project/pycharm-python35/fastText/test360/result/fasttext_trick5_change.csv'
    # change_data = pd.read_csv(change_file, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])
    file_save = '/home/batista/project/project/pycharm-python35/fastText/test360/result/fasttext_trick5.csv'
    # data[['id', 'label']].to_csv(file_save, header=False, index=False)
    with open(old_file, 'r', encoding='utf-8') as old:
        lines = old.readlines()
        res = []
        for line in lines:
            res.append(line)
        print('old file is：', len(lines))

    with open(change_file, 'a', encoding='utf-8') as change:
        for line in res:
            change.write(line + '\n')
        print('write over.')

    with open(all_file, 'r', encoding='utf-8') as all:
        all_ = {}
        lines = all.readlines()
        for line in lines:
            all_[line.strip().split('\t')[0]] = None
        print(all_)

    with open(change_file, 'r', encoding='utf-8') as change:
        lines = change.readlines()
        final_res = []
        for line in lines:
            id_ = line.strip().split(',')[0]
            print(id_)
            if id_ in all_:
                final_res.append(line)
    with open(file_save, 'w', encoding='utf-8') as save:
        i = 0
        for line in final_res:
            save.write(line)
            i += 1
        print('finished and i is:', i)
# merge_change_and_old_data()




fasttext = '/home/batista/project/project/pycharm-python35/fastText/test360/result/'
def process4_new():
    print('process data...')
    testfile = './BDCI2017-360/evaluation_public.tsv'
    data = pd.read_csv(testfile, sep='\\t', header=None, encoding='utf-8', names=['id', 'title', 'content'])


    with open(testfile, 'r', encoding='utf-8') as file:
        res = {}
        for line in file.readlines():
            str_list = line.strip('\n').split('\t')
            res[str_list[0]] = str_list[1] + ' ' + str_list[2]

    with open(fasttext+'0.81299.csv', 'r', encoding='utf-8') as file:
        res_save = []
        label_ = ''
        for line in file.readlines():
            str_list = line.strip('\n').split(',')
            if is_neg2(res[str_list[0]]):
                label_ = 'NEGATIVE'
            else:
                label_ = str_list[1]
            res_save.append(str_list[0] + ',' + label_)
    with open(fasttext+'trick6_my.csv', 'w', encoding='utf-8') as file: # old和change都进行了trick5
        for line in res_save:
            file.write(line + '\n')
    print('finish!')
# process4_new()

# ---- yilonghao -----
import codecs
def trick_yilonghao(pretrain_csv, input_csv, output_csv):
    results = {}

    PATTERN = re.compile(u'\{[\sA-Za-z:0-9\-_;\.]{2,}[\u4E00-\u9FA5]+[\s\S]*\}')
    KEYS = ('pgc', 'text-decoration', 'none', 'outline', 'display', 'block', 'width',
            'height', 'solid', 'position', 'relative', 'padding', 'absolute', 'background',
            'top', 'left', 'right', 'cover', 'font-size', 'font', 'overflow', 'bold',
            'hidden', 'inline', 'block', 'align', 'center', 'transform', 'space', 'vertical',
            'color', 'webkit', 'translatntent')

    # =======================================================================================

    with codecs.open(pretrain_csv, 'r', encoding='utf8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    # 特殊字串
    def is_neg2(x):
        str_1 = '用微信扫描二维码分享至好友和朋友圈'
        m1 = re.findall(str_1, x)  # 2次连续
        return len(m1) >= 3  # 1为13237，>1为13548
    for n, line in enumerate(lines):
        if is_neg2(line[2]):
            results[line[0]] = 'NEGATIVE'

    for n, line in enumerate(lines):
        try:
            if line[2] == '':
                results[line[0]] = 'NEGATIVE'
            elif '获悉、' in line[2]:
                results[line[0]] = 'NEGATIVE'
            elif '但' == line[2][0]:
                results[line[0]] = 'NEGATIVE'
            elif '，' == line[2][-1]:
                results[line[0]] = 'NEGATIVE'
            elif "，的" in line[2] and "，的确" not in line[2] and "，的士" not in line[2] \
                    and "，的哥" not in line[2] and "，的的确确" not in line[2]:
                results[line[0]] = 'NEGATIVE'
            elif "。的" in line[2] and "。的确" not in line[2] and "。的士" not in line[2] \
                    and "。的哥" not in line[2] and "。的的确确" not in line[2]:
                results[line[0]] = 'NEGATIVE'
            elif "也，" in line[2] or '也。' in line[2]:
                results[line[0]] = 'NEGATIVE'
            elif "，了" in line[2] and "，了解" not in line[2] and "，了不起" not in line[2] \
                    and "，了无" not in line[2] and "，了却" not in line[2] and '，了不起' not in line[2]:
                results[line[0]] = 'NEGATIVE'
            elif "。了" in line[2] and "。了解" not in line[2] and "。了不起" not in line[2] \
                    and "。了无" not in line[2] and "。了却" not in line[2] and '。了不起' not in line[2]:
                results[line[0]] = 'NEGATIVE'
        except:
            pass

    for n, line in enumerate(lines):
        key_num = 0
        for key in KEYS:
            if key in line[2]:
                key_num += 1
        if key_num >= 1:
            if re.search(PATTERN, line[2].replace(' ', '')):
                results[line[0]] = 'NEGATIVE'

    # 删除完全相同的文章, 如果两篇文章完全相同, 判定文章为正标签
    content_pool = {}
    for n, line in enumerate(lines):
        if line[2] != '':
            if line[2] not in content_pool:
                content_pool[line[2]] = [n, line[0]]
            else:
                results[line[0]] = 'POSITIVE'
                id_dup = content_pool[line[2]][1]  # 取出与之重复的那篇文章的 id
                results[id_dup] = 'POSITIVE'

    print('change result numbers：', len(results)) # 90212

    with open(input_csv, 'r', encoding='utf-8') as pretrain_file:
        pretrain_res = {}
        for line in pretrain_file.readlines():
            str_list = line.strip('\n').split(',')
            pretrain_res[str_list[0]] = str_list[1]
    with open(output_csv, 'w', encoding='utf-8') as output_file:
        final_results = pretrain_res.copy()
        final_results.update(results)
        for key in final_results:
            output_file.write('%s,%s\n' % (key, final_results[key]))
    diff = 0
    for key in final_results:
        if final_results[key] != pretrain_res[key]:
            diff += 1
    print('diff numbers:', diff) # 2623 - 2649
                                  # 2495

trick_yilonghao(pretrain_csv='./BDCI2017-360/BDCI2017-360-Semi/evaluation_public.tsv',
                input_csv=fasttext+'semi_test_predict1_res.csv',
                output_csv=fasttext+'semi_test_predict1_res_trick1.csv')

# 0.62282 -> 0.62717
# 增加fasttext训练效果0.61143，反而降低，说明过拟合 或者没训练好