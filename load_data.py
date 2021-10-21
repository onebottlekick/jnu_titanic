import csv
import random
import numpy as np


def process(txt):
    if txt == '':
        txt = False
    elif txt == 'male':
        txt = 0
    elif txt == 'female':
        txt = 1
    return float(txt)


def to_one_hot(X):
    n_X = int(np.max(X) + 1)
    return np.eye(n_X)[X.astype(np.int)]


class DataReader:
    def __init__(self, test_ratio=0.2):
        self.test_ratio = test_ratio
        self.X_train, self.X_test, self.y_train, self.y_test = self.read_data()

        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)


    def read_data(self):        
        with open('dataset/train.csv', 'r') as f:
            data = []
            f.readline()
            file = csv.reader(f)
            for line in file:
                # Pclass = process(line[2])
                Sex = process(line[4])
                Age = process(line[5])
                # Sibsp = process(line[6])
                # Parch = process(line[7])
                # Fare = process(line[9])
                Survived = process(line[1])

                data.append((Sex, Age, Survived))
            
            random.shuffle(data)
            data = np.asarray(data)
            norm_data = data/np.max(data, axis=0)
            
            num_raw = norm_data.shape[0]

            X_train = norm_data[:int(num_raw*(1 - self.test_ratio)), :-1]
            y_train = to_one_hot(data[:int(num_raw*(1 - self.test_ratio)), -1])
            X_test = norm_data[int(num_raw*(1 - self.test_ratio)):, :-1]
            y_test = to_one_hot(data[int(num_raw*(1 - self.test_ratio)):, -1])
            
        return X_train, X_test, y_train, y_test


