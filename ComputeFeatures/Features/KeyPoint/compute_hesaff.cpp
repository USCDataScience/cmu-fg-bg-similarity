/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "locker.hpp"
#include "hesaff/pyramid.h"
#include "hesaff/helpers.h"
#include "hesaff/affine.h"
#include "hesaff/siftdesc.h"

using namespace cv;
using namespace std;

struct HessianAffineParams
{
  float threshold;
  int   max_iter;
  float desc_factor;
  int   patch_size;
  bool  verbose;
  HessianAffineParams()
  {
    threshold = 16.0f/3.0f;
    max_iter = 16;
    desc_factor = 3.0f*sqrt(3.0f);
    patch_size = 41;
    verbose = false;
  }
};

int g_numberOfPoints = 0;
int g_numberOfAffinePoints = 0;

struct Keypoint
{
  float x, y, s;
  float a11,a12,a21,a22;
  float response;
  int type;
  unsigned char desc[128];
};

struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
  const Mat image;
  SIFTDescriptor sift;
  vector<Keypoint> keys;
  public:
  AffineHessianDetector(const Mat &image, const PyramidParams &par, const AffineShapeParams &ap, const SIFTDescriptorParams &sp) : 
    HessianDetector(par), 
    AffineShape(ap), 
    image(image),
    sift(sp)
  {
    this->setHessianKeypointCallback(this);
    this->setAffineShapeCallback(this);
  }

  void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
  {
    g_numberOfPoints++;
    findAffineShape(blur, x, y, s, pixelDistance, type, response);
  }

  void onAffineShapeFound(
      const Mat &blur, float x, float y, float s, float pixelDistance,
      float a11, float a12,
      float a21, float a22, 
      int type, float response, int iters) 
  {
    // convert shape into a up is up frame
    rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);

    // now sample the patch
    if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
    {
      // compute SIFT
      sift.computeSiftDescriptor(this->patch);
      // store the keypoint
      keys.push_back(Keypoint());
      Keypoint &k = keys.back();
      k.x = x; k.y = y; k.s = s; k.a11 = a11; k.a12 = a12; k.a21 = a21; k.a22 = a22; k.response = response; k.type = type;
      for (int i=0; i<128; i++)
        k.desc[i] = (unsigned char)sift.vec[i];
      // debugging stuff
      if (0)
      {
        cout << "x: " << x << ", y: " << y
          << ", s: " << s << ", pd: " << pixelDistance
          << ", a11: " << a11 << ", a12: " << a12 << ", a21: " << a21 << ", a22: " << a22 
          << ", t: " << type << ", r: " << response << endl; 
        for (size_t i=0; i<sift.vec.size(); i++)
          cout << " " << sift.vec[i];
        cout << endl;
      }
      g_numberOfAffinePoints++;
    }
  }

  void exportKeypoints(ostream &out)
  {
    out << 128 << endl;
    out << keys.size() << endl;
    for (size_t i=0; i<keys.size(); i++)
    {
      Keypoint &k = keys[i];

      float sc = AffineShape::par.mrSize * k.s;
      Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);
      SVD svd(A, SVD::FULL_UV);

      float *d = (float *)svd.w.data;
      d[0] = 1.0f/(d[0]*d[0]*sc*sc);
      d[1] = 1.0f/(d[1]*d[1]*sc*sc);

      A = svd.u * Mat::diag(svd.w) * svd.u.t();

      out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1);
      for (size_t i=0; i<128; i++)
        out << " " << int(k.desc[i]);
      out << endl;
    }
  }
};

int main(int argc, char **argv)
{
  if (argc < 3) {
    cerr << "Usage: ./prog <indir> <fileslist> <outdir>" << endl;
    return -1;
  }
  string indir = argv[1];
  string fileslist = argv[2];
  string outdir = argv[3];

  // read all images into list
  vector<string> imnames;
  ifstream fin(fileslist.c_str());
  string line;
  while (getline(fin, line)) {
    imnames.push_back(line);
  }
  fin.close();

  #pragma omp parallel for
  for (size_t i = 1; i <= imnames.size(); i++) { // IMP: 1 indexed
    Mat tmp = imread(indir + "/" + imnames[i - 1]);
    string outfpath = outdir + "/" + to_string((long long)i) + ".hesaff.sift";
    if (!Locker::lock(outfpath)) {
      continue;
    }

    Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));

    float *out = image.ptr<float>(0);
    unsigned char *in  = tmp.ptr<unsigned char>(0); 

    for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
    {
      *out = (float(in[0]) + in[1] + in[2])/3.0f;
      out++;
      in+=3;
    }

    HessianAffineParams par;
    double t1 = 0;
    {
      // copy params 
      PyramidParams p;
      p.threshold = par.threshold;

      AffineShapeParams ap;
      ap.maxIterations = par.max_iter;
      ap.patchSize = par.patch_size;
      ap.mrSize = par.desc_factor;

      SIFTDescriptorParams sp;
      sp.patchSize = par.patch_size;

      AffineHessianDetector detector(image, p, ap, sp);
      t1 = getTime(); g_numberOfPoints = 0;
      detector.detectPyramidKeypoints(image);
      cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfAffinePoints << " affine shapes in " << getTime()-t1 << " sec." << endl;

      ofstream out(outfpath.c_str());
      detector.exportKeypoints(out);
    }
    Locker::unlock(outfpath);
  }
}
