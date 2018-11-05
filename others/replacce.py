import os
import json


# def replace_word():
#     for file in os.listdir(r'C:\PicTs\01\outputs'):
#         if os.path.splitext(file)[1] == '.json':
#             with open(file, 'r') as load_file:
#                 load_dict = json.load(load_file)
#                 load_dict['path'] = 'C:/PicTs/01/outputs/' + os.path.splitext(file)[0] + '.jpg'
#
#             with open(file, 'w') as dump_file:
#                 json.dump(load_dict, dump_file)

def deal_txt(path):
    # file_object = open(path)
    # file_content = file_object.read()
    # file_split = file_content.splitlines()
    # print file_split[1]

    # with open(path) as f:
    #     line = f.readline()
    #     while line:
    #         print line
    #         line = f.readline()

    # with open(path) as f:
    #     for line in f.readlines():
    #         print line

    fp = open('./test1.txt', 'w')
    for line in open(path).readlines():
        fp.write(line.replace('wangchao', 'wenlei'))
    fp.close()


if __name__ == '__main__':
    file_path = './test.txt'
    deal_txt(file_path)
