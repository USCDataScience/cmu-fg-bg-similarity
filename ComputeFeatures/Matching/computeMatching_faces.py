#!/usr/bin/python3.4

import numpy as np
import argparse
import os
import scipy.spatial
import operator
from functools import reduce
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgslist', type=str, required=True,
            help='Images list')
    parser.add_argument('-f', '--featuresdir', type=str, required=True,
            help='Directory with all the features as files')
    parser.add_argument('-o', '--outputdir', type=str, required=True,
            help='Directory where ouput will be stored')
    args = parser.parse_args()
    FEAT_DIR = args.featuresdir
    OUT_DIR = args.outputdir
    f = open(args.imgslist)
    imgsList = f.readlines()
    imgsList = list(map(lambda x: x.strip(), imgsList))
    imgsList = imgsList

    files = [f for f in os.listdir(FEAT_DIR) if os.path.isfile(os.path.join(FEAT_DIR, f))]
    files = sorted(files)
    files = files
    nFiles = len(files)
    files_dict = dict(zip(files, range(nFiles)))

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    # find the shape by reading one file
    fpath = os.path.join(FEAT_DIR, files[0])
    nDim = np.size(np.loadtxt(fpath, dtype=float, delimiter='\n'))
    feats = np.empty((nFiles, nDim))

    i = 0
    for fname in files:
        fpath = os.path.join(FEAT_DIR, fname)
        feats[i, :] = np.loadtxt(fpath, dtype=float, delimiter='\n')
        i += 1
    print('Read files')
    Y = scipy.spatial.distance.pdist(feats, 'cosine')
    Y = scipy.spatial.distance.squareform(Y)

    # now combine the face matching outputs
    nImgs = len(imgsList)
    dists = np.empty([nImgs, nImgs])
    for i in range(nImgs):
        fname = imgsList[i]
        for j in range(nImgs):
            fname2 = imgsList[j]
            dists[i][j] = computeDist(fname, fname2, files, files_dict, Y)
    np.savetxt(os.path.join(OUT_DIR, 'dist.txt'), dists, fmt='%.4f')

def computeDist(fname1, fname2, files, files_dict, Y):
    subImgs1 = getSubImgs(fname1, files)
    subIdx1 = list(map(lambda x: files_dict[x], subImgs1))
    subImgs2 = getSubImgs(fname2, files)
    subIdx2 = list(map(lambda x: files_dict[x], subImgs2))
    if len(subIdx1) == 0 or len(subIdx2) == 0:
        return float('Inf')
    sub = Y[list(map(lambda x: [x], subIdx1)), subIdx2]
    return reduce(operator.mul, np.amin(sub, 1), 1) # return product of min dists

def getSubImgs(fname, files):
    # files is a list of file names, fname is one image.
    # return the sub-list of files which satisfies fname_*.jpg
    fbase, _ = os.path.splitext(fname)
    return [name for name in files if re.match(fbase + '_.*\.txt', name)]

if __name__ == '__main__':
    main()

