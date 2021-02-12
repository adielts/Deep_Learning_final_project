import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# the function choose the parameter of c that produce the best accuracy
def best_c(X_train, X_test, y_train, y_test):
    max_accuracy = 0
    max_accuracy_c = 0
    accuracy_arr = []
    c_arr = []
    for i in range(-3, 5):
        logreg = LogisticRegression(penalty='l2', C=10 ** i, max_iter=len(X_train))
        c_arr.append(10 ** i)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_c = 10 ** i
        accuracy_arr.append(accuracy)

    # comparing c plot
    plt.plot(c_arr, accuracy_arr, color='b')
    plt.title("comparing c")
    plt.xlabel("c")
    plt.ylabel("accuracies")
    # plt.show()

    return max_accuracy_c


if __name__ == '__main__':
    # load and normalize the data
    df = pd.read_csv('persons_heart_data.csv')
    labels = df.columns.values
    df = np.asarray(df)
    print(df)
    X = df[:, :-1]
    y = df[:, -1:]
    y = y.flatten()
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

    # choose best c
    best_c = best_c(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    print("best c: " + str(best_c))

    # logistic regression
    logreg = LogisticRegression(penalty='l2', C=best_c, max_iter=len(X_train))
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy_regular_LR = metrics.accuracy_score(y_test, y_pred)
    print(accuracy_regular_LR)
