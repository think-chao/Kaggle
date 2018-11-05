import os
import json

for file in os.listdir(r'C:\PicTs\01\outputs'):
    if os.path.splitext(file)[1] == '.json':
        with open(file, 'r') as load_file:
            load_dict = json.load(load_file)
            load_dict['path'] = 'C:/PicTs/01/outputs/'+os.path.splitext(file)[0]+'.jpg'

        with open(file, 'w') as dump_file:
            json.dump(load_dict, dump_file)

