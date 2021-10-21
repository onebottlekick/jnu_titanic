import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def data():    
    train_data = pd.read_csv('./dataset/train.csv')
    submit_data = pd.read_csv('./dataset/test.csv')
    label = train_data.Survived.to_numpy()
    id_num = submit_data.PassengerId.to_numpy()

    selected_features = ['Sex', 'Age', 'Pclass', 'Fare']

    train_data = train_data[selected_features]
    submit_data = submit_data[selected_features]

    # feature process
    label_en = LabelEncoder()
    train_data.Sex = label_en.fit_transform(train_data.Sex)
    submit_data.Sex = label_en.transform(submit_data.Sex)

    # null process
    train_data = normalize(train_data.interpolate(method='linear', limit_direction='forward'))
    submit_data = normalize(submit_data.interpolate(method='linear', limit_direction='forward'))

    X = train_data.to_numpy()
    submit_X = submit_data.to_numpy()

    # label process
    onehot_en = OneHotEncoder()
    y = onehot_en.fit_transform(label.reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test, submit_X, id_num

