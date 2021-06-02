from sklearn.svm import SVC
import joblib
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def svc_train(train_vecs, y_train, test_vecs, y_test):
    # 创建SVC模型
    cls = SVC(kernel="rbf", verbose=True)
    # 训练模型
    cls.fit(train_vecs, y_train)
    # 保存模型
    joblib.dump(cls, "D:/PycharmProjects/sentiment_analysis/data/svcmodel.pkl")
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
