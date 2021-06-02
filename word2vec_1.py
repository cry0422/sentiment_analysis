import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def build_vector(text, size, wv):
    # 创建一个指定大小的数据空间
    vec = np.zeros(size).reshape((1, size))
    # count是统计有多少词向量
    count = 0
    # 循环所有的词向量进行求和
    for w in text:
        try:
            vec += wv[w].reshape((1, size))
            count += 1
        except:
            continue

    # 循环完成后求均值
    if count != 0:
        vec /= count
    return vec


def get_train_vecs(x_train, x_test):
    # 初始化模型和词表
    wv = Word2Vec(size=300, min_count=2)
    wv.build_vocab(x_train)
    # 训练并建模
    wv.train(x_train, total_examples=1, epochs=1)
    # 获取train_vecs
    train_vecs = np.concatenate([build_vector(z, 300, wv) for z in x_train])
    # 保存处理后的词向量
    np.save('D:/PycharmProjects/sentiment_analysis/data/train_vecs1.npy', train_vecs)
    # 保存模型
    wv.save("D:/PycharmProjects/sentiment_analysis/data/model4.pkl")

    wv.train(x_test, total_examples=1, epochs=1)
    test_vecs = np.concatenate([build_vector(z, 300, wv) for z in x_test])
    np.save('D:/PycharmProjects/sentiment_analysis/data/test_vecs1.npy', test_vecs)


def get_data():
    train_vecs = np.load("D:/PycharmProjects/sentiment_analysis/data/train_vecs1.npy")
    y_train = np.load("D:/PycharmProjects/sentiment_analysis/data/y_train.npy")
    test_vecs = np.load("D:/PycharmProjects/sentiment_analysis/data/test_vecs1.npy")
    y_test = np.load("D:/PycharmProjects/sentiment_analysis/data/y_test.npy")
    return train_vecs, y_train, test_vecs, y_test


def pca(train_vecs, test_vecs):
    pca = PCA(n_components=2)
    pca.fit(train_vecs)
    new_vecs = pca.fit_transform(train_vecs)
    pca1 = PCA(n_components=2)
    pca1.fit(test_vecs)
    new_vecs_test = pca1.fit_transform(test_vecs)
    return new_vecs, new_vecs_test


def svc_train(train_vecs, y_train, test_vecs, y_test):
    # 创建SVC模型
    cls = SVC(kernel="rbf", verbose=True)
    # 训练模型
    cls.fit(train_vecs, y_train)
    # 保存模型
    joblib.dump(cls, "D:/PycharmProjects/sentiment_analysis/data/svcmodel_1.pkl")
    # 预测
    test_result = cls.predict(test_vecs)
    # 输出评分
    # print(cls.score(test_vecs, y_test))
    # true = np.sum(test_result == y_test)
    auc_score = roc_auc_score(y_test, test_result)
    print(auc_score)
    fpr, tpr, thresholds = roc_curve(y_test, test_result)
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 1)  ##设定x轴的范围
    plt.ylim(0.0, 1.1)  ## 设定y轴的范围
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Postive Rate')
    plt.plot(fpr, tpr, linewidth=2, linestyle="-", color='red')
    plt.show()

def word2vec(x_train, x_test):
    get_train_vecs(x_train, x_test)
    train_vecs, y_train, test_vecs, y_test = get_data()
    new_vecs, new_vecs_test = pca(train_vecs, test_vecs)
    svc_train(new_vecs, y_train, new_vecs_test, y_test)
