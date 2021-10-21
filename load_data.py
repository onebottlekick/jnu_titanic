import csv
import random
import numpy as np


def preprocess_sex(txt):
    if txt == 'male':
        txt = 0
    elif txt == 'female':
        txt = 1
    return float(txt)


def preprocess_age(txt):
    if txt == '':
        txt = 29.6
    return float(txt)


# one-hot encode
def to_one_hot(X):
    n_X = int(np.max(X) + 1)
    return np.eye(n_X)[X.astype(np.int)]


class DataReader:
    def __init__(self, kind, test_ratio=0.2):
        self.test_ratio = test_ratio
        self.kind = kind

        if self.kind == 'train':
            self.X_train, self.X_test, self.y_train, self.y_test = self.read_data()        
            print('X_train shape:', self.X_train.shape)
            print('X_test shape:', self.X_test.shape)
            print('y_train shape:', self.y_train.shape)
            print('y_test shape:', self.y_test.shape)

        if self.kind == 'test':
            self.id_num, self.data = self.read_data()


    def read_data(self):        
        with open(f'dataset/{self.kind}.csv', 'r') as f:
            data = []
            id_num = []
            f.readline()
            file = csv.reader(f)

            # process assignment data
            if self.kind == 'test':
                for line in file:
                    PassengerId = int(line[0])
                    # Pclass = process(line[2])
                    Sex = preprocess_sex(line[3])
                    Age = preprocess_age(line[4])
                    # Sibsp = process(line[6])
                    # Parch = process(line[7])
                    # Fare = process(line[9])
                    data.append([Sex, Age])
                    id_num.append(PassengerId)
                    
                return id_num, data

            # process train data
            for line in file:
                # Pclass = process(line[2])
                Sex = preprocess_sex(line[4])
                Age = preprocess_age(line[5])
                # Sibsp = process(line[6])
                # Parch = process(line[7])
                # Fare = process(line[9])
                Survived = float(line[1])
                data.append((Sex, Age, Survived))
                
            random.shuffle(data)
            data = np.asarray(data)
            norm_data = data/np.max(data, axis=0)
            num_row = norm_data.shape[0]
            X_train = norm_data[:int(num_row*(1 - self.test_ratio)), :-1]
            y_train = to_one_hot(data[:int(num_row*(1 - self.test_ratio)), -1])
            X_test = norm_data[int(num_row*(1 - self.test_ratio)):, :-1]
            y_test = to_one_hot(data[int(num_row*(1 - self.test_ratio)):, -1])

        return X_train, X_test, y_train, y_test

