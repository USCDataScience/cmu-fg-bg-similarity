import os
import sys
import BaseHTTPServer
from SocketServer import ThreadingMixIn

# import custom POST handler class
import ExamplePOSThandler

class ThreadedHTTPServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass

class CMU_Generic_Handler( BaseHTTPServer.BaseHTTPRequestHandler ):
    server_version= "CMU_Generic_Handler 0.1"
    def do_POST( self ):
        try:
          response = self.response_handler.post(self)
        except Exception, exc : # catch *all* exceptions
          e = sys.exc_info()[0]
          print exc
          self.send_error(599,"Internal Error: %s" % e)

def run(handler_class=CMU_Generic_Handler, server_address = ('', 8888), service_port = 5555, type='full'):
    print 'Loading resources...'
    MyPostHandler = ExamplePOSThandler.a_POST_handler(service_port, type) # initialize response handler class here
    handler_class.response_handler = MyPostHandler
    srvr = ThreadedHTTPServer(server_address, handler_class)
    print 'Launching server...'
    try:
        srvr.serve_forever() # serve_forever
    except KeyboardInterrupt:
        srvr.socket.close()

if __name__ == "__main__":
    type = sys.argv[1]
    if type == 'full':
      run_on_port = 8900
      service_port = 5568
    elif type == 'fg':
      run_on_port = 8901
      service_port = 5569
    else:
      print 'INVALID options'

    run(server_address = ('', run_on_port), service_port = service_port, type=type)
