import jieba
import re
import xlwt
import jieba.posseg as pseg


# 定义函数创建停用词列表
def stopwordslist(filepath2):
    # 以行的形式读取停用词表，同时转换为列表
    stopword = [line.strip() for line in open(filepath2, 'r', encoding='utf-8').readlines()]
    noword = [line.strip() for line in open('D:/PycharmProjects/sentiment_analysis/data/negation.txt', 'r+', encoding='utf-8').readlines()]
    for word in stopword:
        if word in noword:
            stopword.remove(word)
    degree = [line.strip() for line in open('D:/PycharmProjects/sentiment_analysis/data/adverb of degree.txt', 'r+', encoding='gbk').readlines()]
    new_degree = []
    for i in degree:
        degreeword = i.split(',')
        new_degree.append(degreeword[0])
    for j in stopword:
        if j in new_degree:
            stopword.remove(j)
    return stopword


# 创建数据集并去空
def comments_list(filepath1):
    comments = [line.strip() for line in open(filepath1, 'r', encoding='utf-8').readlines()]
    for line in comments:
        if line == '':
            comments.remove(line)
    return comments


def pretext(contents, filepath2):
    text = []
    # 这里加载停用词的路径
    stopwords = stopwordslist(filepath2)

    for i in range(len(contents)):
        # 去除空格
        content1 = str(contents[i]).replace(" ", "")

        # 只保留中文，去掉符号
        pattern = re.compile("[^\u4e00-\u9fa5]")
        content2 = re.sub(pattern, '', content1)

        # 精确模式分词
        cutwords = jieba.lcut(content2, cut_all=False)

        words = ''
        # for循环遍历分词后的每个词语
        for word in cutwords:
            # 判断分词后的词语是否在停用词表内
            if word not in stopwords:
                if word != '\t':
                    words += word
                    words += "/"
        # 去掉文本中的斜线
        content3 = words
        text.append(content3)

        # 使用for循环逐一获取划分后的词语进行词性标注
        # lastword = pseg.lcut(content3)
        # print('\n【对去除停用词后的分词进行词性标注：】' + '\n')
        # print([(words.word, words.flag) for words in lastword])  # 转换为列表
    # 去重
    text = list(set(text))
    # 去空
    for i in text:
        if i == '':
            text.remove(i)
    for i, e in enumerate(text):
        if e[-1] == '/':
            text[i] = text[i][:-1]
    return text



def storedata(filepath3, text):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    for i in range(len(text)):
        sheet1.write(i, 0, text[i])
    f.save(filepath3)


def pre_process():
    filepath1 = 'D:/PycharmProjects/sentiment_analysis/data/pos.txt'
    filepath2 = 'D:/PycharmProjects/sentiment_analysis/data/cn_stopwords.txt'
    filepath3 = 'D:/PycharmProjects/sentiment_analysis/data/pos_cutwords.xls'
    filepath4 = 'D:/PycharmProjects/sentiment_analysis/data/neg.txt'
    filepath5 = 'D:/PycharmProjects/sentiment_analysis/data/neg_cutwords.xls'

    comments = comments_list(filepath1)
    text = pretext(comments, filepath2)
    storedata(filepath3, text)

    comments1 = comments_list(filepath4)
    text1 = pretext(comments1, filepath2)
    storedata(filepath5, text1)
