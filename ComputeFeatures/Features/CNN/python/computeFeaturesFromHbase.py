import sys, os
caffe_root = '../external/caffe/'
sys.path.insert(0, caffe_root + '/python')
import caffe
sys.path.insert(0, '../external/DiskVector/python')
import PyDiskVectorLMDB
import happybase
import base64
import skimage.io
from StringIO import StringIO
import numpy as np
import time
import operator

def convertJPEGb64ToCaffeImage(img_data_coded, color):
  black_image = np.zeros((256,256,3))
  try:
    img_data = base64.b64decode(img_data_coded)
    img = skimage.io.imread(StringIO(img_data))
  except:
    return black_image
  if reduce(operator.mul, np.shape(img)) == 0:
    return black_image
  # inspired from caffe.io.load_image
  try:
    img = skimage.img_as_float(img).astype(np.float32)
  except:
    print 'Unable to convert image to float32. Returning black image'
    img = black_image
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    if color:
      img = np.tile(img, (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img

def loadCaffeModels():
  MODEL_FILE = os.path.join('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/run_scripts/deploy_pool5.prototxt')
  PRETRAINED = os.path.join('/home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
  mean_image = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
  print np.shape(mean_image)
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
      mean=mean_image,
      channel_swap=(2,1,0), raw_scale=255,
      image_dims=(256,256))
  return net

# @return: a nImgs x 9216D numpy array
def extractPool5Features(imgs, model, normalize = False):
  nImgs = len(imgs)
  features = model.predict(imgs, oversample = False)
  features = np.squeeze(np.reshape(features, (nImgs, -1, 1, 1)))
  if normalize:
    row_norms = np.linalg.norm(features, 2, axis=1)
    features = features / row_norms[:, np.newaxis]
  return features

def readList(fpath):
  f = open(fpath)
  res = f.read().splitlines()
  f.close()
  return res

def getImagesFromIds(ids, hbasetable):
  imgs = []
  for i in ids:
    imgs.append(convertJPEGb64ToCaffeImage(hbasetable.row(i)['image:orig'], True))
  return imgs

def saveFeat(feat, id, stor):
  f = PyDiskVectorLMDB.FeatureVector()
  for i in range(np.shape(feat)[0]):
    f.append(float(feat[i]))
  stor.Put(id, f)

# assumes start_pos 0 indexed
# o/p is also 0 indexed
def getUniqImgIdsList(fpath, start_pos):
  with open(fpath) as f:
    uniqornot = [el[0] for el in f.read().splitlines()[start_pos : ]]
  lno = start_pos
  res = []
  for i in uniqornot:
    if i == 'U':
      res.append(lno)
    lno += 1
  return res

# uniqImIds is 0 indexed
def runFeatExt(imgslist, uniqImIds, model, hbasetable, stor, normalize = False):
  batchSize = model.blobs['data'].num
  batches = [uniqImIds[i:i+batchSize] for i in range(0, len(uniqImIds), batchSize)]
  bid = 1
  for batch in batches:
    print('Doing for %s, batch (%d / %d)' %(imgslist[batch[0]], bid, len(batches)))
    start_time = time.time()
    batch_imglist = []
    for el in batch:
      batch_imglist.append(imgslist[el])
    imgs = getImagesFromIds(batch_imglist, hbasetable)
    loadImg_time = time.time()
    feats = extractPool5Features(imgs, model, normalize)
    featExt_time = time.time()
    # save the feats
    j = 0
    for i in batch:
      saveFeat(feats[j, :], (i + 1) * 10000 + 1, stor)
      j += 1
    save_time = time.time()
    print('Done in \n\tTotal: %d msec\n\tLoad: %d\n\tFeatExt: %d\n\tSave: %d' 
        % ((save_time - start_time) * 1000, 
           (loadImg_time - start_time) * 1000, 
           (featExt_time - loadImg_time) * 1000, 
           (save_time - featExt_time) * 1000))
    bid += 1

def main():
  caffe.set_mode_gpu()
  conn = happybase.Connection('10.1.94.57')
  tab = conn.table('roxyscrape')
  model = loadCaffeModels()
  stor = PyDiskVectorLMDB.DiskVectorLMDB('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/Features/pool5_normed', False)
  imgslist = readList('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/lists/Images.txt')
  #start_pos = 2617800 # default = 0, 0 indexed
  # start_pos is the position in the Images.txt file (0 indexed)
  start_pos = 9140196 # default = 0, 0 indexed
  uniqImIds = getUniqImgIdsList('/home/rgirdhar/memexdata/Dataset/processed/0004_IST/lists/Uniq_sha1.txt', start_pos)
  runFeatExt(imgslist, uniqImIds, model, tab, stor, normalize=True)

if __name__ == '__main__':
  main()


