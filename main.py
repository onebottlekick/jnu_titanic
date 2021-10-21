from load_data import DataReader
from nn import MultiLayerNet
from plot_train_history import plot
from optimizers import *


data = DataReader(test_ratio=0.2)

X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

model = MultiLayerNet(input_size=2, hidden_size_list=[16, 32, 64, 64, 64], output_size=2, use_batchnorm=True)

history = model.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, iter_num=1000000, batch_size=32, optimizer=Adam(lr=0.2))

plot(train_loss=history['train_loss'], train_acc=history['train_acc'], test_acc=history['test_acc'])
