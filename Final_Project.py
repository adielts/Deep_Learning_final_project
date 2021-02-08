import numpy as np


if __name__ == '__main__':
    df = np.genfromtxt('persons_heart_data.csv', delimiter=',')
    print(df)
    array = np.zeros(2)