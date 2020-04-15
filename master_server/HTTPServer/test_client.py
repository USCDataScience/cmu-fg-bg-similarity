#!/usr/bin/env python
import sys
import urllib2
import urllib
import base64

from threading import Thread

def make_request(url):
    data = "http://10.3.2.61/~ubuntu/projects/001_backpage/dataset/corpus/ImagesTexas/Texas_2012_10_10_1349853753000_5_1.jpg"
    data = "http://10.3.2.61/~ubuntu/projects/001_backpage/dataset/corpus/ImagesNevada/Nevada_2014_6_22_1403493916000_5_0.jpg"
    data = "http://10.3.2.61/~ubuntu/projects/001_backpage/dataset/corpus/ImagesNevada/Nevada_2012_10_2_1349231418000_8_5.jpg"
    headers = { 'Content-type' : 'text',  'Content-length' : str(len(data) + 2)}
    req = urllib2.Request(url, data, headers) #POST request
    base64string = base64.encodestring('darpamemex:darpamemex')
    req.add_header("Authorization", "Basic %s" % base64string)
    try:
      response = urllib2.urlopen(req)
      result = response.read()
      print result
    except urllib2.URLError, err:
      print err

def main():
    try:
      # make_request("https://cmu.memexproxy.com:%d" % port)
      make_request("https://cmu.memexproxy.com/segment")
    except urllib2.HTTPError, err:
      print err

main()
