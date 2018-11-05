# -*- coding:utf8 -*-
import pandas as pd
import os

labels = pd.read_csv(r'E:\code\rookie\dog_breed\data\labels.csv')

breed = labels['breed'].unique()
breed_count = len(breed)
print breed_count
breed_dict = {}
record = {}

train_dir = 'E:/code/rookie/dog_breed/data/train/'
val_dir = 'E:/code/rookie/dog_breed/data/val/'
val_path = os.listdir(val_dir)
train_path = os.listdir(train_dir)

# 建立不同种类的狗对应的索引字典
for i in range(len(breed)):
    breed_dict[breed[i]] = i
for i in range(len(labels)):
    labels['breed'][i] = breed_dict[labels['breed'][i]]
    record[labels['id'][i]] = labels['breed'][i]

with open('val.txt', 'a+') as f:
    for val in val_path:
        f.write((str(val) + ' ' + str(record[val.split('.')[0]]) + '\n'))

with open('train.txt', 'a+') as f:
    for train in train_path:
        f.write((str(train) + ' ' + str(record[train.split('.')[0]]) + '\n'))

# with open('label1.txt', 'a+') as f:
#     for line in labels.values:
#         f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))

