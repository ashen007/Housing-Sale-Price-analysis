import numpy as np
import pandas as pd


class LinearRegression_GD:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = 0
        self.cost = []

    def fit(self, x, y):
        ones = np.ones([x.shape[0], 1])
        theta = np.zeros(x.shape[1] + 1)
        x = np.hstack([np.asarray(x), ones])
        y = np.asarray(y)

        for i in range(self.epochs):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.dot(x.T, loss) / len(x)
            theta = theta - gradient * self.learning_rate
            cost = np.sum((np.dot(x, theta) - y) ** 2) / 2 * len(y)
            self.cost.append(cost)

        self.theta = theta
        return theta

    def predict(self, x):
        ones = np.ones([x.shape[0], 1])
        x = np.hstack([np.asarray(x), ones])

        return np.dot(x, self.theta)


class L1:
    def __init__(self, learning_rate, epochs, alpha=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.theta = 0
        self.cost = []

    def fit(self, x, y):
        ones = np.ones([x.shape[0], 1])
        theta = np.zeros(x.shape[1] + 1)
        x = np.hstack([np.asarray(x), ones])
        y = np.asarray(y)

        for i in range(self.epochs):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = (np.dot(x.T, loss) + self.alpha * theta) / len(x)
            theta = theta - gradient * self.learning_rate
            cost = (np.sum((np.dot(x, theta) - y) ** 2) + self.alpha * np.sum(theta ** 2)) / 2 * len(y)
            self.cost.append(cost)

        self.theta = theta
        return theta

    def predict(self, x):
        ones = np.ones([x.shape[0], 1])
        x = np.hstack([np.asarray(x), ones])

        return np.dot(x, self.theta)
