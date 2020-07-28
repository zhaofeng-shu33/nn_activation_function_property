# we use SGD to minimize the object function
import numpy as np
import logging

from sklearn.base import BaseEstimator
import tensorflow as tf

logging.getLogger('tensorflow').disabled = True

class QuasiLinearRegression(BaseEstimator):
    def __init__(self):
        self.epsilon = 0.05
        self.train_time = 1000
        self.order = 3
        self.batch_size = 50
    def fit(self, X, Y):
        X, Y = self._validate_data(X, Y)
        k = X.shape[1]
        self._build_model(k)
        self._train_model(X, Y)
    def _activate_function(self, x):
        return tf.add(x, tf.multiply(tf.constant(self.epsilon,
                      dtype=tf.float64),
                      tf.pow(x, tf.constant(self.order, dtype=tf.float64))))

    def _build_model(self, k):
        self.x = tf.placeholder(tf.float64, shape=(None, k))
        self.y = tf.placeholder(tf.float64, shape=(None, 1))
        w = tf.get_variable('w', [k, 1], dtype=tf.float64, initializer=tf.zeros_initializer)    
        b = tf.get_variable('b', [1], dtype=tf.float64, initializer=tf.zeros_initializer)
        unactivated_term = tf.matmul(self.x, w) + b
        self.y_pred = self._activate_function(unactivated_term)
        self.model = tf.reduce_sum(tf.square(tf.subtract(self.y_pred, self.y)))

    def _train_model(self, X, Y):
        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(self.model)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        n = X.shape[0]
        for _ in range(self.train_time): # epochs
            for i in range(0, n, self.batch_size): # batch size = 1
                i_end = np.min([n, i + self.batch_size])
                x = X[i:i_end, :].reshape((-1, X.shape[1]))
                y = Y[i:i_end].reshape((-1, 1))
                feed_dict = {self.x: x, self.y: y}
                _, loss_value = self.sess.run((train, self.model), feed_dict)
        return loss_value
    def predict(self, X):
        feed_dict = {self.x: X}
        y_pred_value = self.sess.run(self.y_pred, feed_dict)
        return y_pred_value.reshape((y_pred_value.shape[0]))