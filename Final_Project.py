import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


# load, normalize and split the data
def read_normalize_and_split_data():
    df = pd.read_csv('persons_heart_data.csv')
    labels = df.columns.values
    df = np.asarray(df)
    X = df[:, :-1]
    y = df[:, -1:]
    y = y.flatten()
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=5)

    return X_train, X_test, Y_train, Y_test, labels


# the function choose the parameter of c that produce the best accuracy
def best_c(X_train, X_test, Y_train, Y_test):
    max_accuracy = 0
    max_accuracy_c = 0
    accuracy_arr = np.zeros(8)
    c_arr = np.zeros(8)
    for i in range(-3, 5):
        logreg = LogisticRegression(penalty='l2', C=10 ** i, max_iter=len(X_train))
        c_arr[i + 3] = 10 ** i
        logreg.fit(X_train, Y_train)
        Y_pred = logreg.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, Y_pred)
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
    plt.title('Accuracy depends C')
    plt.show()

    return max_accuracy_c


def Logistic_Regression(X_train, X_test, Y_train, Y_test):
    logreg = LogisticRegression(penalty='l2', C=best_c, max_iter=len(X_train))
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    accuracy_Logistic_Regression = metrics.accuracy_score(Y_test, Y_pred)
    print("accuracy Logistic Regression: " + str(accuracy_Logistic_Regression))
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum()))
    return accuracy_Logistic_Regression


def Gaussian_Naive_Bayes(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()
    Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    accuracy_Gaussian_Naive_Bayes = metrics.accuracy_score(Y_test, Y_pred)
    print("accuracy Gaussian Naive Bayes: " + str(accuracy_Gaussian_Naive_Bayes))
    print("Number of mislabeled points out of a total %d points : %d" %(X_test.shape[0], (Y_test != Y_pred).sum()))
    return accuracy_Gaussian_Naive_Bayes


def Random_Forest(X_train, X_test, Y_train, Y_test):
    rm = RandomForestClassifier(n_estimators=20, oob_score=True, n_jobs=-1, random_state=101, max_features=None,
                                min_samples_leaf=20)
    rm.fit(X_train, Y_train)
    Y_pred = rm.predict(X_test)
    acc = rm.score(X_test, Y_test)
    print("accuracy Random Forest: ", acc)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != Y_pred).sum()))
    return acc


def comparing_algorithms(accuracy_Logistic_Regression, accuracy_Gaussian_Naive_Bayes, accuracy_Random_Forest):
    # comparing accuracy plot
    algorithms = ['Logistic Regression', 'Gaussian Naive Bayes', 'Random Forest']
    accuracies = [accuracy_Logistic_Regression, accuracy_Gaussian_Naive_Bayes, accuracy_Random_Forest]
    xpos = np.arange(len(algorithms))
    plt.title("comparing accuracy")
    plt.xlabel("algorithms")
    plt.ylabel("accuracies")
    plt.bar(xpos, accuracies)
    plt.xticks(xpos, algorithms)
    plt.show()


if __name__ == '__main__':
    # load, normalize and split the data
    X_train, X_test, Y_train, Y_test, labels = read_normalize_and_split_data()

    # choose best c
    best_c = best_c(X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy())
    print("best c: " + str(best_c))

    # Logistic Regression
    accuracy_Logistic_Regression = Logistic_Regression(X_train, X_test, Y_train, Y_test)

    # Gaussian Naive Bayes
    accuracy_Gaussian_Naive_Bayes = Gaussian_Naive_Bayes(X_train, X_test, Y_train, Y_test)

    # Random Forest
    accuracy_Random_Forest = Random_Forest(X_train, X_test, Y_train, Y_test)

    # comparing between algorithms
    comparing_algorithms(accuracy_Logistic_Regression, accuracy_Gaussian_Naive_Bayes, accuracy_Random_Forest)