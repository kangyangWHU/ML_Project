from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from utils import get_data_set

# 训练朴素贝叶斯模型

def NB_train(train_vecs, y_train, test_vecs, y_test):
    gnb = GaussianNB()
    gnb.fit(train_vecs, y_train)
    # joblib.dump(gnb, storedpaths + 'model_gnb.pkl')
    test_scores = gnb.score(test_vecs, y_test)
    return test_scores


def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf', gamma=1/3.)
    clf.fit(train_vecs,y_train)
    # save model
    test_scores=clf.score(test_vecs,y_test)
    return test_scores


# 训练决策树模型

def decision_tree(train_vecs, y_train, test_vecs, y_test, depth):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    clf.fit(train_vecs, y_train)
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


# LogisticRegression

def LR_classifier(train_vecs, y_train, test_vecs, y_test, C):
    clf = LogisticRegression(C=C, multi_class='multinomial', penalty='l2', solver='sag', max_iter=1000)
    clf.fit(train_vecs, y_train)
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


def knn_classifier(train_vecs, y_train, test_vecs, y_test, k):
    clf = KNeighborsClassifier(k, weights='uniform')
    clf.fit(train_vecs, y_train)
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


def MP_classifier(train_vecs, y_train, test_vecs, y_test):
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), max_iter=1000)
    clf.fit(train_vecs, y_train)
    test_scores = clf.score(test_vecs, y_test)
    return test_scores


if __name__=='__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = get_data_set("data/waimai10k/waimai10k")

    acc = svm_train(x_train, y_train, x_test, y_test)
    print(acc)
