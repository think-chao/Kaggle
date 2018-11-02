# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
import os, random, shutil
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P


def move_file(file_dir, tar_dir):
    path_dir = os.listdir(file_dir)  # 取图片的原始路径
    file_number = len(path_dir)
    if file_number < 1:
        return
    # rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    # pick_number = int(file_number * rate)  # 按照rate比例从文件夹中取一定数量图片
    pick_number = min(200, file_number)
    sample = random.sample(path_dir, pick_number)  # 随机选取picknumber数量的样本图片
    print (sample)
    for name in sample:
        shutil.move(file_dir + name, tar_dir + name)
    return


# 因为使用imshow将一个矩阵显示为RGB图片，需要
# 将三个32*32的矩阵合成一个32*32*3的三维矩阵
def show_img(data, num):
    red = data[num][0].reshape(1024, 1)
    green = data[num][1].reshape(1024, 1)
    blue = data[num][2].reshape(1024, 1)
    pic = np.hstack((red, green, blue))
    pic_rgb = pic.reshape(32, 32, 3)
    plt.imshow(pic_rgb)
    plt.show()


def convert_lmdb(X, Y, path):
    m = X.shape[0]
    map_size = X.nbytes * 10
    env = lmdb.open(path, map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(m):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()
            datum.label = int(Y[i])
            str_id = '{:08}'.format(i)
    env.close()


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_name.append(file)
    return list_name


if __name__ == '__main__':
    train_dir = 'E:/code/rookie/dog_breed/data/train/'
    val_dir = 'E:/code/rookie/dog_breed/data/val/'

    # move_file(train_dir, val_dir)
    # sub_dir = listdir(img_dir, [])
    # for sub in sub_dir:
    #     move_file(img_dir+sub+'/', sub_img_dir)



