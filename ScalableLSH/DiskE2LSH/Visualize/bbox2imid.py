#!/usr/bin/python

inpdir = '../tempdata/matches/'
featcountfile = '../tempdata/feat_counts.txt'
outdir = '../tempdata/matches2/'

def row2imid(rownum, featcounts):
  imid = 0;
  while rownum > featcounts[imid]:
    rownum = rownum - featcounts[imid]
    imid = imid + 1
  featid = rownum - 1
  imid = imid + 1
  return (imid, featid)

def writeOut(fpath, lst):
  f = open(fpath, 'w')
  for el in lst:
    f.write(str(el) + '\n')
  f.close()

def main():
  f = open(featcountfile)
  perImg = [int(s) for s in f.read().splitlines()]
  f.close()
  for i in range(1, 237 + 1):
    f = open(inpdir + str(i) + '.txt')
    imgs = []
    idxs = []
    for el in [int(s) for s in f.read().splitlines()]:
      [im, idx] = row2imid(el, perImg)
      imgs.append(im)
      idxs.append(idx)
    writeOut(outdir + str(i) + '.txt', imgs)
    writeOut(outdir + str(i) + '_posn.txt', idxs)

if __name__ == '__main__':
  main()

