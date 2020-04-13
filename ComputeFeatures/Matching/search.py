#!/usr/bin/python3.4

import numpy as np
import argparse
import os
import scipy.spatial
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--featuresdir', type=str, required=True,
            help='Directory with all the features as files')
    parser.add_argument('-o', '--outputdir', type=str, required=True,
            help='Directory where ouput will be stored')
    parser.add_argument('-t', '--testfile', type=str, required=True,
            help='File with test images. Each line is class/imagename')
    args = parser.parse_args()
    FEAT_DIR = args.featuresdir
    OUT_DIR = args.outputdir
    TEST_FILE = args.testfile

    pwd = os.getcwd()
    os.chdir(FEAT_DIR)
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]
#    files = files[1:100] ## for debugging
    os.chdir(pwd)
    nFiles = len(files)

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    # find the shape by reading one file
    fpath = os.path.join(FEAT_DIR, files[0])
    nDim = np.size(np.loadtxt(fpath, dtype=float, delimiter='\n'))
    feats = np.empty((nFiles, nDim))

    i = 0
    name2id = {}
    for fname in files:
        fpath = os.path.join(FEAT_DIR, fname)
        feats[i, :] = np.loadtxt(fpath, dtype=float, delimiter='\n')
        icls, iname = getClassAndName(fpath)
        if icls not in name2id.keys():
            name2id[icls] = {}
        name2id[icls][iname] = i
        i += 1
    print('Read files')
    
    # read all the test file names
    fid = open(TEST_FILE)
    testfnames = fid.readlines()
    testfnames = sorted(list(map(lambda x: x.strip(), testfnames)))

    files_np = np.array(files)
    for testfname in testfnames:
        print('Doing for %s\n' % testfname)
        icls, iname = getClassAndName(testfname)
        try:
            feat_i = feats[name2id[icls][iname], :]
            dists = chunkedDistSearch(feat_i, feats, 'cosine');
        except Exception as e:
            print('Error: ', e.with_traceback(None))
            import pdb
            pdb.set_trace()
            continue
        order = np.argsort(dists)
        if not os.path.exists(os.path.join(OUT_DIR, icls, iname)):
            os.makedirs(os.path.join(OUT_DIR, icls, iname))
        np.savetxt(os.path.join(OUT_DIR, icls, iname, 'top.txt'), 
                files_np[np.array(order)][0:20], fmt='%s', delimiter='\n')

def getClassAndName(fpath):
    # get the class and image name
    path, fname = os.path.split(fpath)
    fname, _ = os.path.splitext(fname)
    _, fclass= os.path.split(path)
    return (fclass, fname)

def chunkedDistSearch(feat_i, feats, dist_type):
    # search in chunked fashion or gives a memory error
    dists = np.empty([1, 0])
    cnt = 0
    LIMIT = 200
    while cnt < np.shape(feats)[0]:
        dists_chunk = scipy.spatial.distance.cdist(feat_i[np.newaxis, :], 
                feats[cnt:cnt + LIMIT, :], 'cosine')
        dists = np.append(dists, dists_chunk)
        cnt = cnt + LIMIT
    return dists

if __name__ == '__main__':
    main()

