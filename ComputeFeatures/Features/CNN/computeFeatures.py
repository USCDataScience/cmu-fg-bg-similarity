#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, errno
import itertools
import sys
import scipy
caffe_root = '/exports/cyclops/software/vision/caffe/'
sys.path.append(caffe_root + 'python')
import caffe
PyOpenCV_ROOT = '/exports/cyclops/software/vision/opencv/lib/python2.7/dist-packages/'
sys.path.append(PyOpenCV_ROOT)
import cv2

##### Some constants ######
## For patches
SPLIT_V = 3 # split the image vertically into this many pieces
SPLIT_H = 3 # split the image horizontally into this many pieces
SEG_OVERLAP_THRESH = 0.30 # select for computing features if segmented (foreground)
                          # part less than this ratio
RATIO_KEEP = 1 # ratio of patches to keep when removing on decreasing fg overlap
###########################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, required=True,
            help='Directory path with all the images to process')
    parser.add_argument('-o', '--outputdir', type=str, required=True,
            help='Output directory')
    parser.add_argument('-s', '--segments', type=str, default='',
            help='''Path to folder that has segmentations. If this is specified,
            will only compute in the background (black) regions of the segmentations''')
    parser.add_argument('-t', '--segment-type', type=str, default='mean',
            help='''Set the method to fill the segmented image (this is only used when -s
            is set). Can be 'mean' (default), 'inpaint'.''')
    parser.add_argument('-d', '--dumpdir', type=str, default='',
            help='Set this flag to path to store test images. Else not dumped.')
    parser.add_argument('-f', '--feature', type=str, default='prediction',
            help='could be prediction/fc7/pool5 etc')
    parser.add_argument('-p', '--pooling-type', type=str, default='max',
            help='specify type of pooling (max/avg)')

    args = parser.parse_args()
    IMGS_DIR = args.imagesdir
    OUT_DIR = os.path.join(args.outputdir, args.feature)
    FEAT = args.feature
    SEGDIR = args.segments
    DUMPDIR = args.dumpdir
    SEGTYPE = args.segment_type
    POOLTYPE = args.pooling_type


    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = os.path.join('/exports/cyclops/work/001_Selfies/001_ComputeFeatures/Features/CNN/deploy.prototxt')
    PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    mean_image = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mean_image_normal = mean_image.swapaxes(0,1).swapaxes(1,2)
    mean_image_normal = mean_image_normal / np.max(mean_image_normal) # since caffe images are double - 0 to 1
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
            mean=mean_image,
            channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

    net.set_phase_test()
    net.set_mode_cpu()

    pwd = os.getcwd()
    os.chdir(IMGS_DIR)
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]
    os.chdir(pwd)

    if not os.path.isdir(OUT_DIR):
        mkdir_p(OUT_DIR)

    count = 0
    for frpath in files:
        count += 1
        fpath = os.path.join(IMGS_DIR, frpath)
        fileBaseName, fext = os.path.splitext(frpath)
        fileBasePath, _ = os.path.split(fileBaseName)
        out_fpath = os.path.join(OUT_DIR, fileBaseName + '.txt')
        lock_fpath = os.path.join(OUT_DIR, fileBaseName + '.lock')

        # create the subdir to save output in
        outRelDir = os.path.join(OUT_DIR, fileBasePath)
        if not os.path.exists(outRelDir):
            mkdir_p(outRelDir)

        if os.path.exists(lock_fpath) or os.path.exists(out_fpath):
            print('\tSome other working on/done for %s\n' % fpath)
            continue
        
        mkdir_p(lock_fpath)
        input_image = [caffe.io.load_image(fpath)]

        # segment the image if required
        if len(SEGDIR) > 0:
            print('\tSegmenting image...')
            input_image = segment_image(input_image, SEGDIR, frpath, mean_image_normal, SEGTYPE)
            if len(DUMPDIR) > 0:
                dumppath = os.path.join(DUMPDIR, fileBasePath)
                mkdir_p(dumppath)
                for i in range(len(input_image)):
                    scipy.misc.imsave(os.path.join(DUMPDIR, 
                        fileBaseName + '_' + str(i) + '.jpg'), input_image[i])

        features = []
        for img in input_image:
            prediction = net.predict([img], oversample=False)
            if FEAT == 'prediction':
                feature = prediction.flat
            else:
                feature = net.blobs[FEAT].data[0]; # Computing only 1 crop, by def is center crop
                feature = feature.flat
            features.append(np.array(feature))

        if POOLTYPE == 'max':
            print("NOTE: Using MAX Pooling over %d features" % len(features))
            feature = np.amax(np.array(features), axis=0) # MAX Pooling all the features
        elif POOLTYPE == 'avg':
            print("NOTE: Using Avg Pooling over %d features" % len(features))
            feature = np.mean(np.array(features), axis=0) # AVG POOLING all features
        else:
            print('Pooling type %s not implemented!' % POOLTYPE)
        np.savetxt(out_fpath, feature, '%.7f')
        
        rmdir_noerror(lock_fpath)
        print 'Done for %s (%d / %d)' % (fileBaseName, count, len(files))

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
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            pass

def segment_image(input_image, segdir, frpath, mean_image, segtype):
    if segtype == 'mean':
        return [segment_image_mean(input_image[0], segdir, frpath, mean_image)]
    elif segtype == 'inpaint':
        return [segment_image_inpaint(input_image[0], segdir, frpath)]
    elif segtype == 'patches':
        return segment_image_patches(input_image[0], segdir, frpath)
    elif segtype == 'patches_sliding':
        return segment_image_patches_sliding(input_image[0], segdir, frpath)
    else:
        sys.stderr.write('SEGTYPE ' + segtype + ' not implemented!\n')

def segment_image_mean(input_image, segdir, frpath, mean_image):
    path = os.path.join(segdir, frpath)
    S = caffe.io.load_image(path)
    mean_w, mean_h, _ = np.shape(mean_image)
    S = caffe.io.resize_image(S, (mean_w, mean_h))
    input_image = caffe.io.resize_image(input_image, (mean_w, mean_h))
    input_image[S != 0] = mean_image[S != 0]
    return input_image

def segment_image_inpaint(input_image, segdir, frpath):
    path = os.path.join(segdir, frpath)
    S = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    input_image = scipy.misc.imresize(input_image, np.shape(S))
    input_image = cv2.inpaint(input_image, S, 5, cv2.INPAINT_NS)
    input_image = input_image.astype(float) / np.max(input_image)
    return input_image

def segment_image_patches_sliding(input_image, segdir, frpath):
    # for now, simply make as many segments, ignore segmentation
    initial_w, initial_h, _ = np.shape(input_image)
    ratio = 256.0 / initial_h
    path = os.path.join(segdir, frpath)
    S = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if ratio > 0 and ratio < 1:
       input_image = caffe.io.resize_image(input_image,
               (round(initial_w * ratio), round(initial_h * ratio)))
       S = scipy.misc.imresize(S,
               (int(round(initial_w * ratio)), int(round(initial_h * ratio))))

    patches = []
    h, w, _ = np.shape(input_image)
    sz = 227;
    for i in range(0, max(h - sz + 1, 1), 8):
        for j in range(0, max(w - sz + 1, 1), 8):
            segPatch = S[i : min(i + sz + 1, h), j : min(j + sz + 1, w)]
            iPatch = input_image[i : min(i + sz + 1, h), j : min(j + sz + 1, w), :]
            overlap = np.count_nonzero(segPatch) * 1.0 / np.size(segPatch)
            patches.append((overlap, iPatch))
    patches.sort(key = lambda x: x[0])
    patches = patches[0 : int(RATIO_KEEP * len(patches))]
    patches = [p[1] for p in patches]
    return patches

def segment_image_patches(input_image, segdir, frpath):
    path = os.path.join(segdir, frpath)
    S = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    input_image = scipy.misc.imresize(input_image, np.shape(S))
    segPatches = split_image(S, SPLIT_V, SPLIT_H)
    imgPatches = split_image(input_image, SPLIT_V, SPLIT_H)
    # return patches that don't overlap with white regions in the seg
    overlap = [np.count_nonzero(segPatch) * 1.0 / np.size(segPatch) for segPatch in segPatches]
    select = [x <= SEG_OVERLAP_THRESH for x in overlap]
    imgPatches = list(itertools.compress(imgPatches, select))
    return imgPatches

def split_image(input_image, nv, nh):
    """
    Splits the image into nv x nh patches
    """
    patches = []
    vertPatches = np.array_split(input_image, nv, axis=1)
    for patch in vertPatches:
        patches += np.array_split(patch, nh, axis=0)
    return patches

if __name__ == '__main__':
    main()

