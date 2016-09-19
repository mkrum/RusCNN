#! /usr/bin/env python2.7

from humanTest import loadData

data = loadData()

batch = data.get_train(100)

images_file = open('ht_images.txt', 'w')

for i in range(len(batch[0])):
    images_file.write(batch[1][i]+'/'+batch[0][i]+'\n')
