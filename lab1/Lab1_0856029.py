import numpy as np
import matplotlib.pyplot as plt
from GenData import GenData

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    return y * (1 - y)

def loss(y_true, y_pred):
    return np.mean(0.5 * (y_true - y_pred)**2)
    
def derivative_loss(y_true, y_pred):
    return (y_true - y_pred)

class layer():
    def __init__(self, fan_in, fan_out, act='sigmoid'):
        self.w = np.random.normal(0, 1, (fan_in+1, fan_out))
        if act == 'sigmoid':
            self.act = sigmoid
            self.der_act = der_sigmoid
        else:
            self.act = ReLU
            self.der_act = der_ReLU
        
        
    def forward(self, x):
        x = np.c_[x, np.ones((x.shape[0],1))]
        self.forward_gradient = x
        self.y = self.act(x @ self.w)
        return self.y
    
    def backward(self, derivative_C):
        self.backward_gradient =  self.der_act(self.y) *  derivative_C
        return self.backward_gradient @self.w[:-1].T

    def update(self, learning_rate):
        self.gradient = self.forward_gradient.T @ self.backward_gradient
        self.w -= learning_rate*self.gradient
        return self.gradient
        
class SimpleNet():
    def __init__(self, dim, learning_rate = 0.1, act='sigmoid'):
        self.learning_rate = learning_rate
        
        if act == 'sigmoid':
            self.act = sigmoid
            self.der_act = der_sigmoid
        else:
            self.act = ReLU
            self.der_act = der_ReLU
        
        self.layers = []
        for fan_in, fan_out in zip(dim, dim[1:] + [0]):
            if fan_out == 0:
                break
            self.layers += [layer(fan_in, fan_out, act)]
            
    @staticmethod
    def plot_result(data, y_true, y_pred):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            y_true:   ground truth to the data
            y_pred: predicted results to the data
        """
        assert data.shape[0] == y_true.shape[0]
        assert data.shape[0] == y_pred.shape[0]

        plt.figure(figsize=(8,4))

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if y_true[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if y_pred[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()
        
        
    def forward(self, x):
        _x = x
        for l in self.layers:
            _x = l.forward(_x)
        return _x
    
    def backward(self, dC):
        _dC = dC
        for l in reversed(self.layers):
            _dC = l.backward(_dC)
            
        gradients = []
        for l in self.layers:
            gradients += [l.update(self.learning_rate)]
        return gradients
    
    def train(self, X, y_true, epochs=100000, threshold=0.01):
        convergence = False
        for i in range(epochs):
            if not convergence:
                y = self.forward(X)
                error = loss(y, y_true)
                self.backward(derivative_loss(y, y_true))

                if error < threshold:
                    print('coverged')
                    convergence = True


            if i%5000 == 0 or (convergence):
                print( '[{:4d}] loss : {:.4f} '.format(i, error))

            if convergence:
                break

# learning rate = 0.1
if __name__ == "__main__":
    data_type = ['XOR', 'Linear']
    for d in data_type:

        X_train, y_train = GenData.fetch_data(d, 60)
        X_test, y_test = GenData.fetch_data(d, 20)

        net = SimpleNet([2,4, 4, 1], learning_rate=0.1)
        net.train(X_train, y_train, threshold=0.0001)

        print('='*15, d, '='*15)

        # Training (data leakage)
        y_pred = net.forward(X_train)
        print('train loss : ', loss(y_pred, y_train))
        print('train accuracy : {:.2f}%'.format(np.count_nonzero(np.round(y_pred) == y_train) * 100 / len(y_pred)))

        print('-' * 30)

        # Testing
        y_pred = net.forward(X_test)
        print('test loss : ', loss(y_pred, y_test))
        print('test accuracy : {:.2f}%'.format(np.count_nonzero(np.round(y_pred) == y_test) * 100 / len(y_pred)))
        SimpleNet.plot_result(X_test, y_test, np.round(y_pred))
