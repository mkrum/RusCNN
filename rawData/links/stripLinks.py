#! /usr/bin/env python2.7

fil = open("pagedata.txt")

data = fil.read()

data = data.split('src=')
links = []
for spl in data:
    linksDat = spl.split('"')
    if len(linksDat) > 1:
        links.append(spl.split('"')[1])

linkFile = open("abstractlinks.txt", 'w')
for link in links:
    linkFile.write(link+'\n')
