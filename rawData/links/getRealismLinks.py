#! /usr/bin/env python2.7

import os
import urllib2

fil = open('realismlinks.txt', 'w')

for i in range(1, 152):
    response = urllib2.urlopen('http://www.artnet.com/galleries/fine-art/contemporary-realist-artworks-for-sale/?q=sort-by-price-high-to-low/'+str(i))
    html = response.read()
    html = html.split('img src=')
    links = []
    for spl in html:
        spl = spl.split('"')
        if len(spl) > 1:
            if spl[1][:13] == 'http://images':
                links.append(spl[1])

    for link in links:
        fil.write(link+'\n')

