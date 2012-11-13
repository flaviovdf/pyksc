#-*- coding: utf8

from _trend import predict

from sklearn.base import ClassifierMixin

class TrendLearner(ClassifierMixin):

    def __init__(self, gamma, theta, num_steps):
        self.gamma = gamma
        self.theta = theta
        self.num_steps = num_steps

    def fit(self, X, y):
        self.R = np.asanyarray(X, dtype=np.float64, order='C')
        self.labels = np.asanyarray(y, dtype=np.float64, order='C')

    def predict(self, X):

        X = np.asanyarray(X, dtype=np.float64, order='C')
        return predict(X, self.R, self.labels, self.gamma, self.theta, 
                self.num_steps)
