# -*- coding:utf8 -*-
"""
实现功能，从目录下的每个子目录中随机抽取一张图片存到目标目录中去
涉及到： 列出目录下 的所有子目录
         拼接目录的路径
         进入目录复制或者移动图片
"""
import os
import shutil, random


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_name.append(file)
    return list_name


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


if __name__ == '__main__':
    img_dir = 'C:/PicTs/'
    sub_img_dir = 'C:/PicT/'
    move_file('C:/pics/', sub_img_dir + '05/')
