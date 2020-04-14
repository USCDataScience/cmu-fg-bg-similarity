import numpy as np
import os

inputdir = '../tempdata/matches2'
outputdir = '../tempdata/tops/'
boxesdir = '../tempdata/selsearch_boxes/'
## Expects _posn files to be 0 indexed
N = 237

def main():
    for i in range(1, N + 1):
        print i
        fpath = os.path.join(inputdir, str(i) + '.txt')
        f = open(fpath)
        tops = f.read().splitlines()
        f.close()
        tops = [int(s) for s in tops]
        tops = np.array(tops)
        bboxes = readBboxes(os.path.join(inputdir, str(i) + "_posn.txt"), boxesdir, tops)

        done = np.zeros(N + 1) # don't print more than 1 of one image
        fout = open(outputdir + str(i) + ".txt", 'w')
        for j in range(np.shape(bboxes)[0]):
            if not done[tops[j]]:
                fout.write('%d %f %f %f %f\n' % (tops[j], bboxes[j][0], bboxes[j][1], bboxes[j][2], bboxes[j][3]))
                done[tops[j]] = 1


def readBboxes(fpath_posn, dpath_act, tops):
    f = open(fpath_posn)
    posns = [int(i) for i in f.read().splitlines()]
    res = []
    idx = 0
    for posn in posns:
        line = getLine(dpath_act + str(tops[idx]) + ".txt", posn)
        res.append([float(i) for i in line.split(',')])
        idx += 1
    f.close()
    return np.array(res)

def getLine(fpath, lno):
    with open(fpath) as f:
        for i, line in enumerate(f):
            if i == lno:
                return line.strip()

if __name__ == '__main__':
    main()

