import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('persons_heart_data.csv')
    labels = df.columns.values
    df = np.asarray(df)
    print(df)
