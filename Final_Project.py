import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import itertools
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest


# load, normalize and split the data
def read_normalize_and_split_data():
    df = pd.read_csv('persons_heart_data.csv')
    labels = df.columns.values
    labels = np.delete(labels, len(labels)-1, 0)
    df = np.asarray(df)
    X = df[:, :-1]
    y = df[:, -1:]
    y = y.flatten()
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)

    # train, validation, and test split
    X_train_validation, X_test, Y_train_validation, Y_test = train_test_split(X, y, test_size=0.2,
                                                                              random_state=5)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_validation, Y_train_validation,
                                                                    test_size=0.30, random_state=5)
    return [X_train, X_validation, X_test, Y_train, Y_validation, Y_test, X_train_validation, Y_train_validation,
            labels]


# the function choose the parameter of c that produce the best accuracy
def best_c(X_train, X_val, Y_train, Y_val):
    max_accuracy = 0
    max_accuracy_c = 0
    accuracy_arr = np.zeros(8)
    c_arr = np.zeros(8)
    for i in range(-3, 5):
        model = LogisticRegression(penalty='l1', solver='saga', C=10 ** i, max_iter=len(X_train))
        c_arr[i + 3] = 10 ** i
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val)
        accuracy = metrics.accuracy_score(Y_val, Y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_c = 10 ** i
        accuracy_arr[i + 3] = accuracy

    # comparing c plot
    clrs = ['grey' if (x < max(accuracy_arr)) else 'green' for x in accuracy_arr]
    plt.bar(np.arange(len(accuracy_arr)), 100 * accuracy_arr, color=clrs)
    plt.xticks(np.arange(len(c_arr)), c_arr)
    plt.ylabel('Accuracies (%)')
    plt.xlabel("C's")
    plt.title('LR Accuracy depends C')
    plt.show()

    return max_accuracy_c


def Logistic_Regression(X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_val, Y_train_val):
    # choose best c
    c = best_c(X_train, X_val, Y_train, Y_val)
    print("best c: " + str(c))

    logreg = LogisticRegression(penalty='l1', solver='saga', C=c, max_iter=len(X_train_val))
    logreg.fit(X_train_val, Y_train_val)
    Y_pred = logreg.predict(X_test)
    accuracy_Logistic_Regression = metrics.accuracy_score(Y_test, Y_pred)
    print("Accuracy Logistic Regression: " + str(accuracy_Logistic_Regression))
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum()))
    build_confusion_matrix(Y_test, Y_pred, "Logistic Regression\'s confusion matrix")
    return accuracy_Logistic_Regression


def Gaussian_Naive_Bayes(X_train_validation, X_test, Y_train_validation, Y_test):
    gnb = GaussianNB()
    Y_pred = gnb.fit(X_train_validation, Y_train_validation).predict(X_test)
    accuracy_Gaussian_Naive_Bayes = metrics.accuracy_score(Y_test, Y_pred)
    print("Accuracy Gaussian Naive Bayes: " + str(accuracy_Gaussian_Naive_Bayes))
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum()))
    build_confusion_matrix(Y_test, Y_pred, 'Gaussian Naive Bayes\'s confusion matrix')
    return accuracy_Gaussian_Naive_Bayes


def Random_Forest(X_train_validation, X_test, Y_train_validation, Y_test):
    # acc_arry = np.zeros(20)
    # rand_values = np.arange(10, 210, 10)
    best_acc_and_ypred = np.array(2)
    # for i in range(20):
    #     rm = RandomForestClassifier(n_estimators=rand_values[i], n_jobs=-1, random_state=106, max_features=None,
    #                                 min_samples_leaf=20)
    #     rm.fit(X_train_validation, Y_train_validation)
    #     Y_pred = rm.predict(X_test)
    #     acc = metrics.accuracy_score(Y_test, Y_pred)
    #     acc_arry[i] = acc
    #     if best_acc_and_ypred[0] < acc:
    #         best_acc_and_ypred = [acc, Y_pred]

    rm = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=106, max_features=None, min_samples_leaf=20)
    rm.fit(X_train_validation, Y_train_validation)
    Y_pred = rm.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_pred)

    # plt.plot(rand_values, acc_arry)
    # plt.ylabel('Accuracy')
    # plt.xlabel("Number of Forests")
    # plt.title('Random Forest results')
    # plt.show()
    print("Accuracy Random Forest: ", acc)
    print("Number of mislabeled points out of a total %d points : %d" % (
        X_test.shape[0], (Y_test != Y_pred).sum()))
    # build_confusion_matrix(Y_test, Y_pred, 'Random Forest confusion matrix')
    return acc


def Support_Vector_Classification(X_train_validation, X_test, Y_train_validation, Y_test):
    svc = SVC(kernel='linear')
    svc.fit(X_train_validation, Y_train_validation)
    Y_pred = svc.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_pred)
    print("Accuracy SVC: ", acc)
    print("Number of mislabeled points out of a total %d points : %d" % (
        X_test.shape[0], (Y_test != Y_pred).sum()))
    return acc


def AdaBoost(X_train_validation, X_test, Y_train_validation, Y_test):
    # max_iterations = 100
    # acc_arr = np.zeros(max_iterations)
    # for i in range(1, max_iterations+1):

    clf = AdaBoostClassifier(n_estimators=60, learning_rate=0.065)
    clf.fit(X_train_validation, Y_train_validation)
    Y_pred = clf.predict(X_test)
    acc_arr = metrics.accuracy_score(Y_test, Y_pred)

    # plt.plot(np.arange(0.01, 1.01, 0.01), acc_arr)
    # plt.ylabel('Accuracy')
    # plt.xlabel("Number of estimators")
    # plt.title('Adaboost')
    # plt.show()

    accuracy_AdaBoost = metrics.accuracy_score(Y_test, Y_pred)
    print("Accuracy AdaBoost: " + str(accuracy_AdaBoost))
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum()))
    build_confusion_matrix(Y_test, Y_pred, "AdaBoost\'s confusion matrix")
    return accuracy_AdaBoost


def RFE_alg(X_train_validation, X_test, Y_train_validation, Y_test):
    acc_arr = np.zeros(np.shape(X_train_validation)[1] + 1)
    for i in range(np.shape(X_train_validation)[1], 1, -1):
        rfecv = RFE(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=106, max_features=None,
                                           min_samples_leaf=20), n_features_to_select=i)
        rfecv.fit(X_train_validation, Y_train_validation)
        print('RFE ranking (', i, '):', rfecv.ranking_)
        rfecv.fit(X_train_validation[:, np.argsort(rfecv.ranking_)[:i]], Y_train_validation)
        score = rfecv.score(X_test[:, np.argsort(rfecv.ranking_)[:i]], Y_test)
        print('Accuracy RFE is: ', score)
        acc_arr[i] = score
    acc_arr = acc_arr[2:]
    plot_graph_acc_vs_n_fetures(np.arange(13, 1, -1), np.flip(acc_arr), 'RFE')


def select_best_k(X_train_validation, X_test, Y_train_validation, Y_test):
    acc_arr = np.zeros(np.shape(X_train_validation)[1] + 1)
    for number_of_featurs in range(13, 0, -1):
        print('*** run RF with best', number_of_featurs, 'selected featurs ***')
        k_best = SelectKBest(k=number_of_featurs).fit(X_train_validation, Y_train_validation)
        max_rank_indexes = np.argsort(-k_best.scores_)[:number_of_featurs]
        acc_arr[number_of_featurs] = Random_Forest(X_train_validation[:, max_rank_indexes], X_test[:, max_rank_indexes],
                                                   Y_train_validation, Y_test)
    plot_graph_acc_vs_n_fetures(np.arange(13, 0, -1), np.flip(acc_arr[1:]), 'Select K-Best')


def plot_graph_acc_vs_n_fetures(X_values, Y_values, title):
    plt.plot(X_values, Y_values)
    plt.gca().invert_xaxis()
    plt.ylabel('Accuracy')
    plt.xlabel("Number of featurs")
    plt.title(title)
    plt.show()


def comparing_algorithms(accuracy_Logistic_Regression, accuracy_AdaBoost, accuracy_Random_Forest):
    # comparing accuracy plot
    algorithms = ['Logistic Regression', 'AdaBoost', 'Random Forest']
    accuracies = [accuracy_Logistic_Regression, accuracy_AdaBoost, accuracy_Random_Forest]
    xpos = np.arange(len(algorithms))
    plt.title("comparing accuracy")
    plt.xlabel("algorithms")
    plt.ylabel("accuracies")
    plt.bar(xpos, accuracies)
    plt.xticks(xpos, algorithms)
    plt.show()


def build_confusion_matrix(Y_test, Y_pred, title):
    confusion_matrix = np.zeros((2, 2))
    Y_test = Y_test.astype(np.int64)
    Y_pred = Y_pred.astype(np.int64)
    for i in range(len(Y_test)):
        confusion_matrix[Y_test[i]][Y_pred[i]] += 1
    sn.heatmap(confusion_matrix, annot=True, xticklabels=np.arange(2), yticklabels=np.arange(2))
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


def filter_feateres_from_X(X_train_validation, X_test, set):
    filtered_X_train_validation = [[0]]*len(X_train_validation)
    filtered_X_test = [[0]] * len(X_test)
    for index in set:
        filtered_X_train_validation = np.hstack((filtered_X_train_validation, X_train_validation[:, index:index+1]))
        filtered_X_test = np.hstack((filtered_X_test, X_test[:, index:index + 1]))
    filtered_X_train_validation = np.delete(filtered_X_train_validation, 0, 1)
    filtered_X_test = np.delete(filtered_X_test, 0, 1)
    return filtered_X_train_validation, filtered_X_test


def findsubsets(array, subsetSize):
    return list(itertools.combinations(array, subsetSize))


def best_subset_features(X_train_validation, X_test, Y_train_validation, Y_test, labels):
    features = np.arange(len(labels))
    accuracies = []
    global_max_accuracy_sets = []
    for i in range(1, len(features)+1):
        max_accuracy = 0
        max_accuracy_set = ()
        sets = findsubsets(features, i)
        for set in sets:
            filtered_X_train_validation, filtered_X_test = filter_feateres_from_X(X_train_validation.copy(), X_test.copy(), set)
            accuracy_Random_Forest = Random_Forest(filtered_X_train_validation, filtered_X_test, Y_train_validation, Y_test)
            if accuracy_Random_Forest > max_accuracy:
                max_accuracy = accuracy_Random_Forest
                max_accuracy_set = set
        accuracies.append(max_accuracy)
        global_max_accuracy_sets.append(max_accuracy_set)
    print(accuracies)
    print(global_max_accuracy_sets)
    print("the best subset features: " + str(global_max_accuracy_sets[np.argmax(accuracies)]))
    print("the accuracy of the best subset features:", max(accuracies))

    plt.plot(np.arange(len(features)+1, 1, -1), np.flip(accuracies))
    plt.gca().invert_xaxis()
    plt.ylabel('Accuracy')
    plt.xlabel("number of features")
    plt.title('Accuracy depends number of features')
    plt.show()


if __name__ == '__main__':
    # load, normalize and split the data
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test, X_train_validation, Y_train_validation, labels = read_normalize_and_split_data()

    # Logistic Regression
    # accuracy_Logistic_Regression = Logistic_Regression(X_train.copy(), X_validation.copy(), X_test.copy(),
    #                                                    Y_train.copy(), Y_validation.copy(), Y_test.copy(),
    #                                                    X_train_validation.copy(), Y_train_validation.copy())

    # Gaussian Naive Bayes
    # accuracy_Gaussian_Naive_Bayes = Gaussian_Naive_Bayes(X_train_validation.copy(), X_test.copy(),
    #                                                      Y_train_validation.copy(), Y_test.copy())

    # Random Forest
    # accuracy_Random_Forest = Random_Forest(X_train_validation.copy(), X_test.copy(), Y_train_validation.copy(),
    #                                        Y_test.copy())

    # Support Vector Classification
    # accuracy_SVC = Support_Vector_Classification(X_train_validation.copy(), X_test.copy(),
    #                                              Y_train_validation.copy(), Y_test.copy())

    # AdaBoost
    # accuracy_AdaBoost = AdaBoost(X_train_validation.copy(), X_test.copy(), Y_train_validation.copy(),
    #                              Y_test.copy())

    # comparing between algorithms
    # comparing_algorithms(accuracy_Logistic_Regression, accuracy_Gaussian_Naive_Bayes, accuracy_Random_Forest)

    # RFE
    # RFE_alg(X_train_validation.copy(), X_test.copy(), Y_train_validation.copy(), Y_test.copy())

    # Select best K features
    # select_best_k(X_train_validation.copy(), X_test.copy(), Y_train_validation.copy(),
    #               Y_test.copy())

    best_subset_features(X_train_validation.copy(), X_test.copy(),
                         Y_train_validation.copy(), Y_test.copy(), labels)
