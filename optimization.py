import os
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
import unittest
import argparse
# This is a sample program to explore the property of non-linear activation
METHOD_NAME = ['False', 'tf.sigmoid', 'tf.tanh', 'tf.nn.relu', 'tf.nn.relu6']
TRAIN_TIMES = 100
n = 3
k = 2
def build_model(x, y, activate=False):
    w = tf.get_variable('w', [k,1], dtype=tf.float64, initializer=tf.zeros_initializer)    
    unactivated_term = tf.matmul(x, w)
    if(activate):
        b = tf.get_variable('b', [1,1], dtype=tf.float64)
        b_const = tf.constant(np.ones([n, 1]))  
        unactivated_term = tf.add(unactivated_term, tf.multiply(b, b_const)) 
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
    '''see https://math.stackexchange.com/questions/444700/uniform-distribution-on-the-surface-of-unit-sphere 
       for detail
    '''
    normal_list = np.random.normal(size=n)
    return normal_list / np.linalg.norm(normal_list)

def generate_uniform_sample():
    spherical_coordinate = get_spherical_coordinate()
    y = tf.constant(spherical_coordinate, shape=(n, 1))
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
    parser.add_argument('--debug', default=False, type=bool, nargs='?', const=True, help='whether to debug') 
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collect', default=False, type=bool, nargs='?', const=True)
    parser.add_argument('--table', default=False, type=bool, nargs='?', const=True)
    args = parser.parse_args()
    TRAIN_TIMES = args.train_times
    n = args.n
    k = args.k
    activate = False
    exec('activate = ' + args.activate)
    if activate:
        k = k -1
    if args.seed != 0:
        np.random.seed(args.seed)
    if args.debug:
        import pdb
        pdb.set_trace()
    if(args.collect):
        collect_results(args.sample_times)
    elif(args.table):
        generate_report_table()
    else:
        average_value = get_average(args.sample_times, activate)
        print(args.activate, average_value)
