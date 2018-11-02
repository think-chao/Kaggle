import pandas as pd
import os

labels = pd.read_csv(r'E:\code\rookie\dog_breed\data\labels.csv')
breed = labels['breed'].unique()
breed_count = len(breed)
print breed_count
breed_dict = {}

train_dir = 'E:/code/rookie/dog_breed/data/train/'
val_dir = 'E:/code/rookie/dog_breed/data/val/'
val_path = os.listdir(val_dir)
print type(val_path[0].split('.')[0])
print type(labels['id'][0])
train_path = os.listdir(train_dir)

for i in range(len(breed)):
    breed_dict[breed[i]] = i
for i in range(len(labels)):
    labels['breed'][i] = breed_dict[labels['breed'][i]]
print labels.head()
print breed_dict

with open('val.txt', 'a+') as f:
    for i, val in enumerate(val_path):
        f.write((str(val) + '\t' + str(breed_dict[val.split('.')[0]]) + '\n'))

with open('label1.txt', 'a+') as f:
    for line in labels.values:
        f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))
