import zmq
import json
import base64
import subprocess
import time
import os
import urllib2

class a_POST_handler:
    def __init__(self,a_number):
        self.number=a_number

    def post(self,request_handler):
        # check request headers
        if not request_handler.headers.has_key('content-length'):
            request_handler.send_error(550,"No content-length given")
        try:
            content_length = int(request_handler.headers['content-length'])
        except ValueError:
            content_length = 0
        if content_length<=0:
            request_handler.send_error(551,"invalid content-length given")

        # if text is given, response appropriately
        if request_handler.headers.has_key('content-type') and request_handler.headers['content-type']=='text':
            text = request_handler.rfile.read(content_length)
            print 'Got Request at %s with %s' % (request_handler.path, text)

            # retrieve the response from other services and return
            port_num = 8889 # full image search service
            if request_handler.path == '/bgsearch':
              port_num = 8891
            elif request_handler.path == '/segment':
              port_num = 8890
            elif request_handler.path == '/IST_fullImg':
              port_num = 8892
            elif request_handler.path == '/IST_bgImg':
              port_num = 8893
            elif request_handler.path == '/IST_fullImg_ITQ':
              port_num = 8894
            elif request_handler.path == '/IST_bgImg_ITQ':
              port_num = 8895
            elif request_handler.path == '/ATF_bgImg_ITQ':
              port_num = 8896
            elif request_handler.path == '/JPLWeapons2_bgImg':
              port_num = 8897
            elif request_handler.path == '/JPLWeapons2_fullImg':
              port_num = 8898
            elif request_handler.path == '/JPLWeapons2_fgImg':
              port_num = 8899
            elif request_handler.path == '/CounterfeitElectrics_fullImg':
              port_num = 8900
            elif request_handler.path == '/CounterfeitElectrics_foreImg':
              port_num = 8901

            resp = self.getResponse(text, port_num)

            self.sendResponse(resp,request_handler)
        else:
            request_handler.send_error(552, "No or unrecognized content-type")

    def sendResponse(self,body,request_handler):
        request_handler.send_response( 200 )
        request_handler.send_header( "content-type", "text" )
        request_handler.send_header( "content-length", str(len(body)) )
        request_handler.end_headers()
        request_handler.wfile.write( body )

    def getResponse(self, text, port_num):
        headers = { 'Content-type' : 'text',  'Content-length' : str(len(text))}
        req = urllib2.Request('http://localhost:%d' % port_num, text, headers)
        try:
            response = urllib2.urlopen(req)
            result = response.read()
            return result
        except urllib2.URLError, err:
            print err

