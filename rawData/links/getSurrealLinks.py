#! /usr/bin/env python2.7

import os
import urllib2

target = open('surreallinks.txt', 'w')
source = open('surrealdata.txt', 'r')

lines = source.readlines()

for line in lines:
    if line[:3] == 'src':
        line = line[4:]
        line = line.split('"')[1]
        target.write(line+'\n')
