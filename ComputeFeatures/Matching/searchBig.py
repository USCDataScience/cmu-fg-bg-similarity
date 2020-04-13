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
    parser.add_argument('-r', '--initial-ranklist-dir', type=str, required=True,
            help='''Initial Ranklist (maybe computed using BoW). If set, will only search
            in files specified in $DIR/fpath/top.txt''')
    parser.add_argument('-n', '--top-n', type=int, default=1000,
            help='''Number of top images to consider for feature matching''')
    parser.add_argument('-s', '--dump-scores', action='store_const', default=False, 
            const=True, help='''Set this to print the scores too in top.txt files''')
    args = parser.parse_args()
    FEAT_DIR = args.featuresdir
    OUT_DIR = args.outputdir
    TEST_FILE = args.testfile
    PRERANKED_DIR = args.initial_ranklist_dir
    TAKE_N = args.top_n
    DUMP_SCORES = args.dump_scores

    pwd = os.getcwd()
    os.chdir(FEAT_DIR)
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]
#    files = files[1:100] ## for debugging
    os.chdir(pwd)
    nFiles = len(files)

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    # read all the test file names
    fid = open(TEST_FILE)
    testfnames = fid.readlines()
    testfnames = sorted(list(map(lambda x: x.strip(), testfnames)))

    for testfname in testfnames:
        print('Doing for %s\n' % testfname)
        icls, iname = getClassAndName(testfname)
        out_fpath = os.path.join(OUT_DIR, icls, iname, 'top.txt')
        if os.path.exists(out_fpath):
            sys.stderr.write('Already done for %s\n' % out_fpath)
            continue

        # check if pre-ranked list exists, and read the select list
        prerank_path = os.path.join(PRERANKED_DIR, icls, iname, 'top.txt')

        if not os.path.exists(prerank_path):
            sys.stderr.write('Preranked files doesnt exist: %s\n\tContinuing..\n' 
                    % prerank_path)
            continue
        # modify the feats and files list being used for this image
        f = open(prerank_path)
        preranked_list = f.readlines()
        preranked_list = sorted(list(map(lambda x: x.strip(), preranked_list)))
        preranked_list = preranked_list[0 : TAKE_N]
        if not preranked_list:
            sys.stderr.write('Preranked List empty for %s\n\tContinuing..\n'
                    % prerank_path)
            continue

        feats = readFeats(FEAT_DIR, preranked_list)
        files_np = np.array(preranked_list)

        try:
            feat_i = readFeat(FEAT_DIR, testfname)
            dists = chunkedDistSearch(feat_i, feats, 'cosine');
        except Exception as e:
            print('Error: ', e.with_traceback(None))
            import pdb
            pdb.set_trace()
            continue
        order = np.argsort(dists)
        if not os.path.exists(os.path.join(OUT_DIR, icls, iname)):
            os.makedirs(os.path.join(OUT_DIR, icls, iname))

        # write output to file
        if DUMP_SCORES:
            np.savetxt(out_fpath, 
                    np.rec.fromarrays((files_np[order][0 : TAKE_N], sorted(dists[0 : TAKE_N])), 
                        names = ('name', 'score')),
                    fmt='%s %.6f', delimiter='\n')
        else:
            np.savetxt(out_fpath, files_np[order][0 : TAKE_N], fmt='%s', delimiter='\n')

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

def readFeats(FEAT_DIR, files):
    feats = []
    for fpath in files:
        feats.append(readFeat(FEAT_DIR, fpath))
    return np.array(feats)

def readFeat(FEAT_DIR, fpath):
    fcls, fname = getClassAndName(fpath)
    path = os.path.join(FEAT_DIR, fcls, fname + '.txt')
    return np.loadtxt(path, dtype=float, delimiter='\n')

if __name__ == '__main__':
    main()

