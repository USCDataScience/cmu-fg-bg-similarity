import zmq
import json
import happybase
import urlparse


class a_POST_handler:
  def __init__(self,a_service_port,type='full'):
    self.service_port=a_service_port
    if type == 'full':
        self.BASE_URL = 'http://localhost:8000/'
    elif type=='fg':
	self.BASE_URL = 'http://localhost:8001/'
    else:
	self.BASE_URL = 'http://localhost:8000/'

  def __del__(self):
    self.conn.close()

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

      # retrieve the result from zmq and return
      resp = self.retrieveMatches(text)
      
      #print('Received response: '+str(resp))

      # convert to JSON
      resp = self.convertToJSON(resp)

      self.sendResponse(resp,request_handler)
    else:
      request_handler.send_error(552, "No or unrecognized content-type")

  def sendResponse(self,body,request_handler):
    request_handler.send_response( 200 )
    request_handler.send_header( "content-type", "text" )
    request_handler.send_header( "content-length", str(len(body)) )
    request_handler.end_headers()
    request_handler.wfile.write( body )

  def retrieveMatches(self, text):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:" + str(self.service_port))
    # strings must be null-delimited (C style)
    socket.send(text + "\0")
    message = socket.recv()
    return message

  def convertToJSON(self, resp):
    matches = [m.split(':') for m in resp.strip().split(',')][:-1]
    #print(matches)
    #print('Size of matches: '+str(len(matches)))
    matches = self.fix_commas(matches)
    matches = [(self.getS3URL(m[1]), float(m[0])) for m in matches]
    return json.dumps(matches)

  def getS3URL(self, imid):
    return urlparse.urljoin(self.BASE_URL, imid)

  def fix_commas(self, matches_list):
    matches = []

    for i in range(0, len(matches_list)):
	m = matches_list[i]
	#print('Match is '+str(m))
	if len(m) == 2:
	  #print('Appending '+str(m))
	  matches.append(m)
	else:
	  #print("match not of size 2")
          z = len(matches)-2
          #print('z:'+str(z))
	  #print('Current matches: '+str(matches))
	  #print('Current matches size: '+str(len(matches)))
	  #print('Matches list z-1 index: ('+str(z)+'): '+str(matches[z-1]))
          #print("M1-1 1: "+str(matches[z-1][1]))
          #print("M0: "+str(m[0]))
	  matches[z-1][1] = matches[z-1][1] + m[0]

    return matches

