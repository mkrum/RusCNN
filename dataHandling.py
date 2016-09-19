#! /usr/bin/env python2.7

import numpy as np
import os
from PIL import Image
from sklearn.cross_validation import train_test_split

class loadData:
    #modified to handled different images based 
    def __init__(self, choose=[1, 1, 1]):
        data = []
        if choose[0]:
            data.append(self.load(0, 'surreal'))
        if choose[1]:
            data.append(self.load(1, 'realism'))
        if choose[2]:
            data.append(self.load(2, 'abstract'))
        imageData   = np.asarray(reduce(lambda x,y: x + y, [ d[0] for d in data ]))
        labels      = np.asarray(reduce(lambda x,y: x + y, [ d[1] for d in data ]))
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(imageData, labels,  test_size=.2)
        self.index = 0

    def load(self, index, arttype):
        files = os.listdir('data/'+arttype)
        label = self.convert_to_onehot(index)
        labels = []
        files_div = self.divide(files, 100)
        images = []
        for fd in files_div:
            images += self.images_to_np(fd, arttype)
        for _ in range(len(images)):
            labels.append(label)
 
        return images, labels


    def divide(self, l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

    def images_to_np(self, files, arttype):
        images = [ Image.open('data/'+arttype+'/'+fil, 'r') for fil in files ]
        self.height, self.width = images[0].size
        imageData =  [ np.asarray(img, dtype=np.float32).ravel() / 255.0  for img in images ] 
        imageData =  [ i for i in imageData if i.shape[0] == 7500 ]
        return imageData

    def convert_to_onehot(self, index):
        label = [0, 0, 0]
        label[index] = 1
        return np.asarray(label)

    def get_train(self, batch_size):
       images = []
       labels = []
       for i in xrange(batch_size):
           images.append(self.train_images[self.index])
           labels.append(self.train_labels[self.index])
           self.index = (self.index + 1) % len(self.train_labels)
       return images, labels

    def get_test(self):
        return self.test_images, self.test_labels
    
    def get_height(self):
        return self.height

    def get_width(self):
        return self.width
