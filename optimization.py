import os
from datetime import datetime
import json
import numpy as np
import scipy
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import unittest
import argparse
# This is a sample program to explore the property of non-linear activation
METHOD_NAME = ['False', 'tf.sigmoid', 'tf.tanh', 'tf.nn.relu', 'tf.nn.relu6', 'cubic', 'quadratic']
TRAIN_TIMES = 100
n = 3
k = 1
epsilon = 0.05

def cubic(x):
    return tf.add(x, tf.multiply(tf.constant(epsilon, dtype=tf.float64), tf.pow(x, tf.constant(3, dtype=tf.float64))))

def quadratic(x):
    return tf.add(x, tf.multiply(tf.constant(epsilon, dtype=tf.float64), tf.pow(x, tf.constant(2, dtype=tf.float64))))

def build_model(x, y, activate=False):
    w = tf.get_variable('w', [k, 1], dtype=tf.float64, initializer=tf.zeros_initializer)    
    unactivated_term_1 = tf.matmul(x, w)
    b = tf.get_variable('b', [1, 1], dtype=tf.float64, initializer=tf.zeros_initializer)
    b_const = tf.constant(np.ones([n, 1]))  
    unactivated_term = tf.add(unactivated_term_1, tf.multiply(b, b_const))
    if activate:
        y_pred = activate(unactivated_term)
    else:
        y_pred = unactivated_term
    loss = tf.reduce_sum(tf.square(tf.subtract(y_pred, y)))
    return loss

def train_model(loss):
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss) 
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)    
    for i in range(TRAIN_TIMES):
        _, loss_value = sess.run((train, loss))
    return loss_value

def get_spherical_coordinate():
    '''actually get normal distribution with sigma^2 = 1/n
    '''
    normal_list = np.random.normal(size=n)
    return normal_list / np.sqrt(n)

def get_orthogonal_coordinate():
    z = scipy.randn(n, n) # n by n random matrix
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph)
    return q[:,:k]

def generate_uniform_sample():
    spherical_coordinate = get_spherical_coordinate()
    y = tf.constant(spherical_coordinate, shape=(n, 1))
    x_np_array = get_orthogonal_coordinate()
    x = tf.constant(x_np_array)
    return (x, y)

def artificial_dataset():
    x = np.random.uniform(0, np.pi, 200)
    y = np.exp(np.sin(x) + np.random.normal(size=200) / 2)
    x = tf.constant(x, shape=(200, 1))
    y = tf.constant(y, shape=(200, 1))
    global n,k
    n = 200
    k = 1
    return (x, y)

def model_run(activate=False, sample_generation=generate_uniform_sample):
    tf.reset_default_graph()
    x_t, y_t = sample_generation()
    loss = build_model(x_t, y_t, activate)
    loss_value = train_model(loss)
    return loss_value

def get_average(num_times, activate=False, sample_generation=generate_uniform_sample):
    total_value = 0
    for _ in range(num_times):
        total_value += model_run(activate, sample_generation)
    return total_value / num_times



def task(method_name, num_times, q):
    activate_inner = eval(method_name)
    average_value = get_average(num_times, activate_inner)
    q.put({method_name: average_value})

def collect_results(num_times):
    time_str = datetime.now().strftime('%Y-%m-%d')
    file_name = '%d-%d-%d-%d-%s.json' %(n, k, num_times, TRAIN_TIMES, time_str)
    dic = {}
    from multiprocessing import Process, Queue
    process_list = []
    q = Queue()
    for i in METHOD_NAME:
        t = Process(target=task, args=(i, num_times, q))
        t.start() 
        process_list.append(t)  
    for i in range(5):
        process_list[i].join()
        dic.update(q.get())
    with open('build/' + file_name,'w') as f:
        json.dump(dic, f,indent=4)    

def generate_report_table():
    from tabulate import tabulate
    _headers = ['mse', 'none', 'sigmoid', 'tanh', 'relu', 'relu6']
    table = []
    for i in os.listdir('build'):
        n_str, k_str = i.split('-')[0], i.split('-')[1]
        table_row = ['n=%s,k=%s' % (n_str, k_str)]
        with open('build/' + i) as f:
            dic = json.load(f)
            for j in METHOD_NAME:
                table_row.append(dic[j])
        table.append(table_row)
    md_table_string = tabulate(table, headers=_headers, tablefmt='github', floatfmt='.3f')
    time_str = datetime.now().strftime('%Y-%m-%d')
    with open('report-%s.md' % time_str, 'w') as f:
        f.write(md_table_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activate', default='False', choices=METHOD_NAME)
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--train_times', type=int, default=100)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collect', default=False, type=bool, nargs='?', const=True)
    parser.add_argument('--table', default=False, type=bool, nargs='?', const=True)
    parser.add_argument('--artificial', default=False, type=bool, nargs='?', const=True)
    args = parser.parse_args()
    TRAIN_TIMES = args.train_times
    n = args.n
    k = args.k
    activate = False
    exec('activate = ' + args.activate)
    if args.seed != 0:
        np.random.seed(args.seed)
    if(args.collect):
        collect_results(args.sample_times)
    elif(args.table):
        generate_report_table()
    else:
        if args.artificial:
            sample_generation_function = artificial_dataset
        else:
            sample_generation_function = generate_uniform_sample
        average_value = get_average(args.sample_times,
                                    activate,
                                    sample_generation=sample_generation_function)
        print(args.activate, average_value)
