import sys, os
import happybase
import base64
from StringIO import StringIO
import numpy as np
import time
import operator
import sha
import time
import base64
from PIL import Image

hash_type = 'dhash'

last_print = time.time()
def tic_toc_print(msg):
  global last_print
  if time.time() > last_print + 1:
    print(msg)
    last_print = time.time()

def readList(fpath):
  f = open(fpath)
  res = f.read().splitlines()
  f.close()
  return res

def saveFeat(feat, fpath):
  f = open(fpath, 'a')
  f.write(feat + '\n')
  f.close()

def extractSha1Hash(img):
  obj = sha.new(img)
  return obj.hexdigest()

def extractDhash(img, hash_size = 16):
  try:
    image = Image.open(StringIO(base64.b64decode(img)))
  except:
    return ''
  # Grayscale and shrink the image in one step.
  try:
    image = image.convert('L').resize(
      (hash_size + 1, hash_size),
      Image.ANTIALIAS,
    )
  except:
    return ''

  pixels = list(image.getdata())

  # Compare adjacent pixels.
  difference = []
  for row in xrange(hash_size):
    for col in xrange(hash_size):
      pixel_left = image.getpixel((col, row))
      pixel_right = image.getpixel((col + 1, row))
      difference.append(pixel_left > pixel_right)

  # Convert the binary array to a hexadecimal string.
  decimal_value = 0
  hex_string = []
  for index, value in enumerate(difference):
    if value:
      decimal_value += 2**(index % 8)
    if (index % 8) == 7:
      hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
      decimal_value = 0

  return ''.join(hex_string)

def readImage(imid, tab):
  return tab.row(imid)['image:orig']

def runFeatExt(imgslist, hbasetable, start_pos):
  cur_pos = start_pos
  while cur_pos < len(imgslist):
    tic_toc_print('Doing for %s (%d / %d)' %(imgslist[cur_pos], cur_pos, len(imgslist)))
    img = readImage(imgslist[cur_pos], hbasetable)
    if hash_type == 'sha1':
      feats = extractSha1Hash(img)
    elif hash_type == 'dhash':
      feats = extractDhash(img)
    saveFeat(feats, '/home/rgirdhar/memexdata/Dataset/processed/0004_IST/Features/DHash/hashes.txt')
    cur_pos += 1

def main():
  conn = happybase.Connection('10.1.94.57')
  tab = conn.table('roxyscrape')
  imgslist = readList('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/lists/Images.txt')
  start_pos = 1256920 # default = 0
  runFeatExt(imgslist, tab, start_pos)

if __name__ == '__main__':
  main()


