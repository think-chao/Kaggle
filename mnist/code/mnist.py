import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import lmdb
import csv

caffe_root = '/home/chao/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

train_data_path = '../data/train.csv'6
test_data_path = '../data/test.csv'

train_data = pd.read_csv(train_data_path).values.astype(np.uint8)
total = train_data.shape[0]
train_x = train_data[:int(total * 0.9), 1:]
train_y = train_data[:int(total * 0.9), 0]
test_x = train_data[int(total * 0.9):, 1:]
test_y = train_data[int(total * 0.9):, 0]

train_x = train_x.reshape((train_x.shape[0], 1, 28, 28))
test_x = test_x.reshape((test_x.shape[0], 1, 28, 28))

im1 = train_x[10, 0]
print train_y[10]
plt.rcParams['image.cmap'] = 'gray'
plt.imshow(im1)
plt.show()


def covert_lmdb(X, y, path):
    m = X.shape[0]
    map_size = X.nbytes * 10  # donot worry , mapsize no harm
    # http://lmdb.readthedocs.io/en/release/#environment-class
    env = lmdb.open(path, map_size=map_size)
    # http://lmdb.readthedocs.io/en/release/#lmdb.Transaction
    with env.begin(write=True) as txn:
        for i in range(m):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tostring()  # tobeytes if np.version.version >1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


train_lmdb_path = '../data/train_lmdb'
test_lmdb_path = '../data/test_lmdb'
covert_lmdb(train_x, train_y, train_lmdb_path)
covert_lmdb(test_x, test_y, test_lmdb_path)

def Alexnet(lmdb, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                             source=lmdb, transform_param=dict(scale=1./255), ntop=2)


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.fc2 = L.InnerProduct(n.relu1, num_output=500, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.fc2, in_place=True)
    n.score = L.InnerProduct(n.relu2, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


with open('train.prototxt', 'w') as f:
    f.write(str(lenet(train_lmdb_path, batch_size=100)))
with open('test.prototxt', 'w') as f:
    f.write(str(lenet(test_lmdb_path, batch_size=100)))


def gen_solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_iter.append(42)
    s.test_interval = 378
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 0.0005
    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75
    s.display = 100
    s.max_iter = 3780
    s.snapshot = 1000
    s.snapshot_prefix = 'lenet'
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    with open('solver.prototxt', 'w') as f:
        f.write(str(s))


gen_solver('train.prototxt', 'test.prototxt')
caffe.set_mode_cpu()
solver = caffe.get_solver('solver.prototxt')
solver.solve()

"""
def gen_deploy():
    n = caffe.NetSpec()
    n.data = L.Input()

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.fc2 = L.InnerProduct(n.relu1, num_output=500, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.fc2, in_place=True)
    n.score = L.InnerProduct(n.relu2, num_output=10, weight_filler=dict(type='xavier'))

    n.prob = L.Softmax(n.score)
    return n.to_proto()


with open('deploy.prototxt', 'w') as f:
    f.write(str(gen_deploy()))
    
"""
