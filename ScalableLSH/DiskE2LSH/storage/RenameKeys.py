import lmdb
import time

# Takes as input a file with old and new IDs, one pair
# per line, space separated
fpath = '/memexdata/Dataset/processed/0001_Backpage/Features/CNN/bg_translate_list/BgTranslateList.txt'
input_fpath = '/memexdata/Dataset/processed/0001_Backpage/Features/CNN/pool5_bg_normed.old'
output_fpath = '/memexdata/Dataset/processed/0001_Backpage/Features/CNN/pool5_bg_normed_renamed'

last_print = time.time()
def tic_toc_print(msg):
  global last_print
  if time.time() > last_print + 1:
    print(msg)
    last_print = time.time()

def readPairs(fpath):
  f = open(fpath, 'r')
  res = [el.split() for el in f.read().splitlines()]
  f.close()
  return res

def main():
  readEnv = lmdb.Environment(input_fpath, readonly=True)
  writeEnv = lmdb.Environment(output_fpath, readonly=False,
      map_size=1000000000000) # 1 TB
  data = readPairs(fpath)
  with readEnv.begin() as readTx:
    for pair in data:
      f = readTx.get(pair[0])
      with writeEnv.begin(write=True) as writeTxn:
        writeTxn.put(pair[1], f)
      tic_toc_print('Done for %s' % pair)

if __name__ == '__main__':
  main()

