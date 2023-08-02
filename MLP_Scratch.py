import numpy as np
# import datasets.fashion_mnist.loader as mnist
from keras.datasets import fashion_mnist
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import pickle



class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.test_costs = []
        self.train_acc = []
        self.test_acc = []

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def ReLU(self,x):
        return np.maximum(0.,x)    

    def ReLU_derivative(self,x):
        return np.greater(x, 0.).astype(np.float32)    

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def initialize_parameters(self, strategy):
        np.random.seed(1)
        if strategy == 'normal':
            for l in range(1, len(self.layers_size)):
                self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(self.layers_size[l - 1])
                self.parameters["b" + str(l)] = np.random.randn(self.layers_size[l], 1)
        elif strategy == 'zeros':
            for l in range(1, len(self.layers_size)):
                self.parameters["W" + str(l)] = np.zeros((self.layers_size[l], self.layers_size[l - 1])) / np.sqrt(self.layers_size[l - 1])
                self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
            
    
        
    def forward(self, X, activation):
        store = {}
        A = X.T

        if activation == "relu":
            for l in range(self.L - 1):
                Z = self.parameters["W" + str(l + 1)].dot(A) + \
                    self.parameters["b" + str(l + 1)]
                A = self.ReLU(Z)
                store["A" + str(l + 1)] = A
                store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
                store["Z" + str(l + 1)] = Z
        
        
        elif activation == "sigmoid":
            for l in range(self.L - 1):
                Z = self.parameters["W" + str(l + 1)].dot(A) + \
                    self.parameters["b" + str(l + 1)]
                A = self.sigmoid(Z)
                store["A" + str(l + 1)] = A
                store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
                store["Z" + str(l + 1)] = Z

        

        Z = self.parameters["W" + str(self.L)].dot(A) + \
            self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def backward(self, X, Y, store, activation):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dZ = A - Y.T

        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            if activation == "relu":
                dZ = dAPrev * self.ReLU_derivative(store["Z" + str(l)])
            elif activation == "sigmoid":
                dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])

            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives


    def fit(self, X_train, Y_train,X_test,Y_test, activation="relu", learning_rate=0.01, n_iterations=2500, strategy="normal"):
        np.random.seed(1)

        self.n = X_train.shape[0]

        self.layers_size.insert(0, X_train.shape[1])

        self.initialize_parameters(strategy)
        for loop in range(n_iterations):
            A, store = self.forward(X_train, activation) 
            A_test, store_test = self.forward(X_test, activation)
            
            cost = -np.mean(Y_train * np.log(A.T + 1e-8))
            cost_test = -np.mean(Y_test * np.log(A_test.T + 1e-8))

            derivatives = self.backward(X_train, Y_train, store, activation)
            derivatives_test = self.backward(X_test, Y_test, store_test, activation)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]
            
            train_acc = self.predict(X_train, Y_train, activation)
            test_acc = self.predict(X_test, Y_test, activation)
                
            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)

            if loop % 10 == 0:                
                print("Train Cost: ", cost, "Train Accuracy:", train_acc)
                print("Test Cost: ", cost_test, "Test Accuracy:", test_acc)
                print()

            # if loop % 10 == 0:
            self.costs.append(cost)
            self.test_costs.append(cost_test)

    def predict(self, X, Y, activation):
        A, cache = self.forward(X, activation)
        self.cost_test = -np.mean(Y * np.log(A.T + 1e-8))
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100

    def plot_cost(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(self.costs)), self.costs, label="train loss")
        plt.plot(np.arange(len(self.test_costs)), self.test_costs, label="test cost")
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(self.train_acc)), self.train_acc, label="train accuracy")
        plt.plot(np.arange(len(self.test_acc)), self.test_acc, label="test accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        

def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # mnist_dataset = load_dataset('mnist')

    train_x, train_y, test_x, test_y = pre_process_data(
        train_x, train_y, test_x, test_y)

    train_x = np.reshape(train_x,(train_x.shape[0], -1))
    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    layers_dims = [50, 10]
    
    # set activation strategy
    activation_list = ["sigmoid", "relu"]
    strategy_list = ["normal", "zeros"]
    
    ann = ANN(layers_dims)

    # for activation in activation_list:
    #     for s in strategy_list:
    ann.fit(train_x, train_y,test_x, test_y, activation_list[1], learning_rate=0.1, n_iterations=100, strategy=strategy_list[0])
    print("Train Accuracy:", ann.predict(train_x, train_y, activation_list[1]))
    print("Test Accuracy:", ann.predict(test_x, test_y, activation_list[1]))
    with open("relu_normal.pkl", "wb") as f:
        pickle.dump(ann, f)
    ann.plot_cost()


    

