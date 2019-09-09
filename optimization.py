import numpy as np
import tensorflow as tf
import unittest
import argparse
# This is a sample program to explore the property of non-linear activation
TRAIN_TIMES = 100
n = 3
k = 2
def build_model(x, y, activate=False):
    w = tf.get_variable('w', [2,1], dtype=tf.float64)
    b = tf.get_variable('b', [3,1], dtype=tf.float64) 
    unactivated_term = tf.add(tf.matmul(x,w),b)
    if(activate):   
        y_pred = activate(unactivated_term)
    else:
        y_pred = unactivated_term
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    return loss

def train_model(loss):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)    
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    for i in range(TRAIN_TIMES):
      _, loss_value = sess.run((train, loss))
    return loss_value  

def get_spherical_coordinate():
    '''see https://math.stackexchange.com/questions/444700/uniform-distribution-on-the-surface-of-unit-sphere 
       for detail
    '''
    normal_list = np.random.normal(n)
    return normal_list / np.linalg.norm(normal_list)

def generate_uniform_sample():
    spherical_coordinate = get_spherical_coordinate()
    y = tf.constant(spherical_coordinate, shape=(n, 1))
    x_random_number_base = np.random.random(2*k)
    x_np_array = np.zeros([k,n])
    for i in range(k):
        x_np_array[i,:] = get_spherical_coordinate()
    x = tf.constant(x_np_array.T)
    return (x, y)

def model_run(activate=False):
    tf.reset_default_graph()
    x_t, y_t = generate_uniform_sample()
    loss = build_model(x_t, y_t, activate)
    loss_value = train_model(loss)
    return loss_value

def get_average(num_times, activate=False):
    total_value = 0
    for i in range(num_times):
        total_value += model_run(activate)
    return total_value / num_times;

class TestFunc(unittest.TestCase):
    def test_get_spherical_coordinate(self):
        arr = get_spherical_coordinate()
        self.assertAlmostEqual(np.linalg.norm(arr), 1.0)
    def test_generate_uniform_sample(self):
        x_t, y_t = generate_uniform_sample()
        self.assertEqual(x_t.shape, (3,2))
        self.assertEqual(y_t.shape, (3,1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activate', default='False')
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--train_times', type=int, default=100)
    args = parser.parse_args()
    TRAIN_TIMES = args.train_times
    activate = False
    exec('activate = ' + args.activate)
    average_value = get_average(args.sample_times, activate)
    print(args.activate, average_value)
