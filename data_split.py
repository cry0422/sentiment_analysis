import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_file_and_preprocessing():
    neg = pd.read_excel("D:/PycharmProjects/sentiment_analysis/data/neg_cutwords.xls", header=None, index=None)
    pos = pd.read_excel("D:/PycharmProjects/sentiment_analysis/data/pos_cutwords.xls", header=None, index=None)
    # 这是两类数据都是x值
    pos['words'] = pos[0].apply(lambda x: list(x.split('/')))
    neg['words'] = neg[0].apply(lambda x: list(x.split('/')))
    # 需要y值  0 代表neg 1代表是pos
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    X = np.concatenate((pos['words'], neg['words']))
    # 切分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    # 保存数据
    np.save("D:/PycharmProjects/sentiment_analysis/data/y_train.npy", y_train)
    np.save("D:/PycharmProjects/sentiment_analysis/data/y_test.npy", y_test)
    return X_train, X_test
