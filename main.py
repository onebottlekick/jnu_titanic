from load_data import data
from nn import MultiLayerNet
from optimizers import *
from plot_train_history import plot
from submission import submit


# load preprocessed data
X_train, X_test, y_train, y_test, submit_X, id_num = data()

# init model
model = MultiLayerNet(input_size=4, hidden_size_list=[64, 256, 64, 512, 32, 64, 128], activation='relu', output_size=2, use_batchnorm=True, use_dropout=True, dropout_ratio=0.2)

# train model
history = model.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, epochs=100, batch_size=32, optimizer=Adam(lr=0.01))

# predict answers
predict = model.predict(np.array(submit_X))
answer = [a.argmax() for a in predict]

# make submission.csv file
submit(id_num, answer)

# plot model history
plot(train_loss=history['train_loss'], train_acc=history['train_acc'], test_acc=history['test_acc'])