from collections import defaultdict


# 载入情感词典，否定词典，程度副词词典
def positive_words_list(filepath1):
    # 以行的形式读取停用词表，同时转换为列表
    positive_list = [line.strip() for line in open(filepath1, 'r+').readlines()]
    return positive_list


def negative_words_list(filepath2):
    negative_list = [line.strip() for line in open(filepath2, 'r+').readlines()]
    return negative_list


def negation_list(filepath3):
    negation_lists = [line.strip() for line in open(filepath3, 'r+', encoding='utf-8').readlines()]
    return negation_lists


def adverb_list(filepath4):
    # 读取程度副词文件
    degree_file = open(filepath4, 'r+', encoding='gbk')
    degree_list = degree_file.readlines()
    degree_dict = defaultdict()
    # 转为程度副词字典对象，key为程度副词，value为对应的程度值
    for d in degree_list:
        degree_dict[d.split(',')[0]] = d.split(',')[1]
    return degree_dict


def classify(contents):
    # 加载词典
    pos_list = positive_words_list('D:/PycharmProjects/sentiment_analysis/data/positive_word.txt')
    neg_list = negative_words_list('D:/PycharmProjects/sentiment_analysis/data/negative_word.txt')
    no_list = negation_list('D:/PycharmProjects/sentiment_analysis/data/negation.txt')
    adv_list = adverb_list('D:/PycharmProjects/sentiment_analysis/data/adverb of degree.txt')

    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    pos_word = dict()
    neg_word = dict()
    no_word = dict()
    degree_word = dict()

    for j in range(len(contents)):
        word = contents[j]
        if word in pos_list and word not in no_list and word not in adv_list.keys() and word not in neg_list:
            pos_word[j] = 1
        elif word in neg_list and word not in no_list and word not in adv_list.keys():
            neg_word[j] = -1
        elif word in no_list and word not in adv_list.keys():
            no_word[j] = -1
        elif word in adv_list.keys():
            degree_word[j] = adv_list[word]
    return pos_word, neg_word, no_word, degree_word


# 计算情感值
def value_compute(pos_word, neg_word, no_word, degree_word, contents):
    if pos_word or neg_word:
        score = 0
        w = 1
        for i in range(len(contents)):
            if i in degree_word.keys():
                w *= float(degree_word[i])
            elif i in no_word.keys():
                w *= -1
            elif i in pos_word.keys():
                score += float(w) * float(pos_word[i])
                w = 1
            elif i in neg_word.keys():
                score += float(w) * float(neg_word[i])
                w = 1
    else:
        score = 0
    return score


# 存储情感值
def store_result(comment, value):
    with open('D:/PycharmProjects/sentiment_analysis/data/value.txt', 'a', encoding='utf-8') as f:
        f.write('/'.join(comment) + ',' + str(value) + '\n')


def score(comment):
    pos_word, neg_word, no_word, degree_word = classify(comment)
    value = value_compute(pos_word, neg_word, no_word, degree_word, comment)
    store_result(comment, value)
    return value
