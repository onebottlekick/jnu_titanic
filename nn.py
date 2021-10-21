from nn_functions import *


class ThreeLayerNet:

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, weight_init_std=0.001):
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['W3'] = weight_init_std*np.random.randn(hidden2_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        
        return y
        

    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])        
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        
        # backward
        dy = (y - t) / batch_num

        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        
    
        da2 = np.dot(dy, W3.T)
        dz2 = sigmoid_grad(a2)*da2
        
        grads['W2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, W2.T)
        dz1 = sigmoid_grad(a1)*da1
        
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    
    def fit(self, X_train, X_test, y_train, y_test, iter_num=1000, batch_size=100, learning_rate=0.25):
        train_size = X_train.shape[0]
        history = {}
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        # iter_per_epoch = max(train_size/batch_size, 1)

        from scipy.special import expit

        for i in range(iter_num):
            batch_mask = np.random.choice(train_size, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]

            grad = self.gradient(X_batch, y_batch)

            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                self.params[key] -= learning_rate * grad[key]

            loss = self.loss(X_batch, y_batch)
            train_loss_list.append(loss)

            # if i%iter_per_epoch == 0:
            if i%100 == 0:
                train_acc = self.accuracy(X_train, y_train)
                test_acc = self.accuracy(X_test, y_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("Step: {:04d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}".format(i+1, loss, train_acc, test_acc))
        print("Step: {:04d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}".format(iter_num, loss, train_acc, test_acc))

        history['train_loss'] = train_loss_list
        history['train_acc'] = test_acc_list
        history['test_acc'] = test_acc_list

        return history

