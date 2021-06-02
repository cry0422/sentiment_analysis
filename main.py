import pre_process
import data_split
import value
import word2vec
import SVM
import numpy as np
import word2vec_1

pre_process.pre_process()
X_train, X_test = data_split.load_file_and_preprocessing()
word2vec_1.word2vec(X_train, X_test)
train_vecs, y_train, test_vecs, y_test = word2vec.vector(X_train, X_test)
score_all = []
score_test = []
for comment in X_train:
    score = value.score(comment)
    score_all.append(score)
for i in range(len(score_all)):
    train_vecs[i] = np.multiply(score_all[i], train_vecs[i])
for comment_test in X_test:
    score1 = value.score(comment_test)
    score_test.append(score1)
for j in range(len(score_test)):
    test_vecs[j] = np.multiply(score_test[j], test_vecs[j])
SVM.svc_train(train_vecs, y_train, test_vecs, y_test)



