import re
import codecs


def pre_evl():
    results = {}

    PATTERN = re.compile(u'\{[\sA-Za-z:0-9\-_;\.]{2,}[\u4E00-\u9FA5]+[\s\S]*\}')
    KEYS = ('pgc', 'text-decoration', 'none', 'outline', 'display', 'block', 'width',
            'height', 'solid', 'position', 'relative', 'padding', 'absolute', 'background',
            'top', 'left', 'right', 'cover', 'font-size', 'font', 'overflow', 'bold',
            'hidden', 'inline', 'block', 'align', 'center', 'transform', 'space', 'vertical',
            'color', 'webkit', 'translatntent')

    # =======================================================================================

    with codecs.open('./BDCI2017-360/evaluation_public_old.tsv', 'r', encoding='utf8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    # 90%
    for line in lines:
        if len(line) == 2:
            line.append('')
        dtl = line[2].replace(' ', '').replace('，', ' 。').replace('、', ' 。').split('。')
        pool = []
        t = 0
        length = 0
        for i in dtl:
            if len(i) < 20:  # 句子长度小于20，跳过
                continue
            if i not in pool:
                pool.append(i)
            else:
                results[line[0]] = 'NEGATIVE'

    # =======================================================================================

    with codecs.open('./BDCI2017-360/evaluation_public.tsv', 'r', encoding='utf8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

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

    print(len(results))

    fw = open('./result/result_part.csv', 'w')
    for key in results:
        fw.write('%s,%s\n' % (key, results[key]))

    for line in lines: # 默认剩下都是pos
        if line[0] not in results:
            fw.write('%s,%s\n' % (line[0], 'POSITIVE'))

if __name__ == '__main__':
    pre_evl()

