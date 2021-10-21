import matplotlib.pyplot as plt
import numpy as np

def plot(train_loss, train_acc, test_acc):
    fig = plt.figure()    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # plot loss history
    x_loss = np.arange(len(train_loss))
    ax1.plot(x_loss, train_loss)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("loss")

    # plot train_accuracy, test_accuracy
    x_acc = np.arange(len(train_acc))
    ax2.plot(x_acc, train_acc, label='train acc')
    ax2.plot(x_acc, test_acc, label='test acc', linestyle='--')
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()