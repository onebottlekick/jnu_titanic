from load_data import DataReader
from nn import MultiLayerNet
from optimizers import *
from plot_train_history import plot
from submission import submit


# load train data
train_data = DataReader(kind='train', test_ratio=0.2)
X_train, X_test, y_train, y_test = train_data.X_train, train_data.X_test, train_data.y_train, train_data.y_test

# init model
model = MultiLayerNet(input_size=2, hidden_size_list=[12, 12, 12], activation='relu', output_size=2, use_batchnorm=True)

# train model
history = model.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, epochs=10, batch_size=32, optimizer=Adam(lr=0.05))

# load assignment data
assignment_data = DataReader(kind='test')

# predict answers
predict = model.predict(np.array(assignment_data.data))
answer = [a.argmax() for a in predict]

# make submission.csv file
submit(assignment_data.id_num, answer)

# plot model history
plot(train_loss=history['train_loss'], train_acc=history['train_acc'], test_acc=history['test_acc'])