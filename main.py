from load_data import DataReader
from nn import *
from plot_train_history import plot


data = DataReader(test_ratio=0.2)

X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

model = ThreeLayerNet(input_size=2, hidden1_size=20, hidden2_size=10, output_size=2)

history = model.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, learning_rate=0.2, iter_num=10000, batch_size=32)

plot(train_loss=history['train_loss'], train_acc=history['train_acc'], test_acc=history['test_acc'])
