#!/usr/bin/python3.4

import numpy as np
import argparse
import os
import scipy.spatial

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--featuresdir', type=str, required=True,
        help='Directory with all the features as files')
parser.add_argument('-o', '--outputdir', type=str, required=True,
        help='Directory where ouput will be stored')
args = parser.parse_args()
FEAT_DIR = args.featuresdir
OUT_DIR = args.outputdir

pwd = os.getcwd()
os.chdir(FEAT_DIR)
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]
os.chdir(pwd)
nFiles = len(files)

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
np.savetxt(os.path.join(OUT_DIR, 'dist.txt'), scipy.spatial.distance.squareform(Y), fmt='%.4f')
np.savetxt(os.path.join(OUT_DIR, 'imagenames.txt'), np.array(files), fmt='%s', delimiter='\n')

