#! /usr/bin/env python2.7

import os

links = open('surreallinks.txt').readlines()

os.chdir('surreal')

for link in links:
    os.system('curl -O '+link)
