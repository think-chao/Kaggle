caffe_root = '/home/chao/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import pandas as pd
deploy_path = './deploy.prototxt'
caffemodel_path = './lenet_iter_3920.caffemodel'
caffe.set_mode_cpu()
clf = caffe.Classifier(deploy_path, caffemodel_path, image_dims=(28, 28))
test = pd.read_csv('../data/test.csv')
test_array = test.values
n = np.shape(test)[0]
a = []
for i in range(n):
    input = test_array[i, :] / 255.
    input = input.reshape((1, 28, 28, 1))
    reslut = clf.predict(input, oversample=False).argmax()
    a.append(reslut)
ImageId = [i + 1 for i in range(n)]
submit_pd = pd.DataFrame({'ImageId': ImageId, 'Label': a})
submit_pd.to_csv('./model.csv', index=False)