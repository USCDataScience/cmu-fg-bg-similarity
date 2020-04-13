#!/usr/bin/python2.7

import numpy as np
import argparse
import os
import scipy.spatial
import time
import errno

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--featuresdir', type=str, required=True,
            help='Directory with all the features as files')
    parser.add_argument('-o', '--outputdir', type=str, required=True,
            help='Directory where ouput will be stored')
    args = parser.parse_args()
    FEAT_DIR = args.featuresdir
    OUT_DIR = args.outputdir
    SCORES_DIR = os.path.join(OUT_DIR, 'match_scores')

    pwd = os.getcwd()
    os.chdir(FEAT_DIR)
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]
    os.chdir(pwd)
    nFiles = len(files)

    mkdir_p(OUT_DIR)
    np.savetxt(os.path.join(OUT_DIR, 'imagenames.txt'), np.array(files), fmt='%s', delimiter='\n')

    # find the shape by reading one file
    fpath = os.path.join(FEAT_DIR, files[0])
    nDim = np.size(np.loadtxt(fpath, dtype=float, delimiter='\n'))

    i = 0

    scores = np.zeros(nFiles)
    for fname in files:
        start_time = time.time()
        OUT_PATH = os.path.join(SCORES_DIR, fname)
        LOCK_PATH = OUT_PATH + '.lock'
        if os.path.exists(LOCK_PATH) or os.path.exists(OUT_PATH):
            print('%s already done\n' % fname)
            continue
        mkdir_p(LOCK_PATH)

        fpath = os.path.join(FEAT_DIR, fname)
        feat1 = np.loadtxt(fpath, dtype=float, delimiter='\n')
        j = 0
        scores[:] = 0
        for fname2 in files:
            fpath2 = os.path.join(FEAT_DIR, fname2)
            feat2 = np.loadtxt(fpath2, dtype=float, delimiter='\n')
            scores[j] = scipy.spatial.distance.cosine(feat1, feat2)
            j += 1

        i += 1
        np.savetxt(OUT_PATH, scores, fmt='%.6f', delimiter='\n')
        rmdir_noerror(LOCK_PATH)
        print('Done for %s (%d / %d) in %s sec\n' % (fname, i, nFiles, time.time() - start_time))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def rmdir_noerror(path):
    try:
        os.rmdir(path)
    except OSError as exc:
        pass

if __name__ == '__main__':
    main()

