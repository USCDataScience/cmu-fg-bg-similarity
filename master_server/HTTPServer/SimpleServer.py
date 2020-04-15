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
      except Exception, exc: # catch *all* exceptions
        print exc
        e = sys.exc_info()[0]
        self.send_error(599,"Internal Error: %s" % e)

def run(handler_class=CMU_Generic_Handler, server_address = ('', 8888), ):
  print 'Loading resources...'
  MyPostHandler = ExamplePOSThandler.a_POST_handler(3) # initialize response handler class here
  handler_class.response_handler = MyPostHandler
  srvr = ThreadedHTTPServer(server_address, handler_class)
  print 'Launching server...'
  try:
      srvr.serve_forever() # serve_forever
  except KeyboardInterrupt:
      srvr.socket.close()

if __name__ == "__main__":
  run()
