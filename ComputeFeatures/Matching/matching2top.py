#!/usr/bin/python3.4
# Code to convert matching scores output (as from computeMatchingBig)
# to top.txt output (as from searchBig.py)

import numpy as np
import argparse
import os
import scipy.spatial
import sys

def main():
    parser = argparse.ArgumentParser()
    ## INPUT DIR must have a 
    parser.add_argument('-d', '--inputdir', type=str, required=True,
            help='Directory with all the scores against all images as files')
    parser.add_argument('-o', '--outputdir', type=str, required=True,
            help='Directory where ouput will be stored')
    parser.add_argument('-i', '--imgslist', type=str, required=True,
            help='File with list of images used for NxN search')
    parser.add_argument('-n', '--topn', type=int, default=1000,
            help='Number of top matches to print')

    args = parser.parse_args()
    IN_DIR = args.inputdir
    OUT_DIR = args.outputdir
    TEST_FILE = args.imgslist
    TOPN = args.topn

    with open(TEST_FILE) as f:
        imgslist = f.read().splitlines()
        imgslist_np = np.array(imgslist)
        n = len(imgslist)
        for img in imgslist:
            with open(os.path.join(IN_DIR, img)) as f2:
                if not f2:
                    sys.stderr.write('Not found: %s.. continuing..\n' % img)
                    continue
                scores = [float(x) for x in f2.readlines()]
                order = np.argsort(np.array(scores))
                scores = sorted(scores)

                outdir = os.path.join(OUT_DIR, os.path.splitext(img)[0])
                outpath = os.path.join(outdir, 'top.txt')
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                np.savetxt(outpath, 
                    np.rec.fromarrays((imgslist_np[order][0 : min(TOPN, n)], 
                        scores[0 : min(TOPN, n)]), 
                        names = ('name', 'score')),
                    fmt='%s %.6f', delimiter='\n')
                print('Done for %s\n' % img)
                

if __name__ == '__main__':
    main()

