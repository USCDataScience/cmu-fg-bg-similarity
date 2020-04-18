/**
 * Author: Rohit Girdhar
 */
#include <opencv2/opencv.hpp>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <leveldb/db.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "hdf5.h"
#include "hdf5_hl.h"

#include <string>
#include <fstream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#define DIM 50
#define OFFSET ((256-227)/2)
#define NW_IMG_WID 256
#define NW_IMG_HT 256


#define METHOD 2
// define METHODS
#define STOR_HDF5 1
#define STOR_JPG 2

using namespace std;
using namespace cv;
using namespace caffe;
using std::vector;
namespace fs = boost::filesystem;

void genSegImg(float xmin, float ymin, float xmax, float ymax, const Mat& S, Mat& res);

void makeParentDir(const string& fpath) {
  fs::path fp(fpath);
  fs::create_directory(fp.parent_path());
}

int CreateDir(const char *sPathName, int beg) {
  char DirName[256];
  strcpy(DirName, sPathName);
  int i, len = strlen(DirName);
  if (DirName[len - 1] != '/')
    strcat(DirName, "/");

  len = strlen(DirName);

  for (i = beg; i < len; i++) {
    if (DirName[i] == '/') {
      DirName[i] = 0;
      if (access(DirName, 0) != 0) {
        CHECK(mkdir(DirName, 0755) == 0)<< "Failed to create folder "<< sPathName;
      }
      DirName[i] = '/';
    }
  }

  return 0;
}


char buf[101000];
int main(int argc, char** argv)
{
  /**
   * argv[1] = prototxt
   * [2] = caffemodel
   * [3] = List of images
   * [4] = output dir
   */

	Caffe::set_phase(Caffe::TEST);

	Caffe::set_mode(Caffe::CPU);
	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
  string fpath_bbox = caffe_test_net.layer_by_name("data_55")->layer_param().transform_param().loc_result();

	vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();

	string labelFile(argv[3]);
	int data_counts = 0;
	FILE * file = fopen(labelFile.c_str(), "r");
	while(fgets(buf,100000,file) > 0)
	{
		data_counts++;
	}
	fclose(file);

	vector<Blob<float>*> dummy_blob_input_vec;
	string rootfolder(argv[4]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);

	int counts = 0;

	ifstream fin(labelFile.c_str());
  ifstream fin_bbox(fpath_bbox.c_str());

	Blob<float>* c1 = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
  int c2 = c1->num();
	int batchCount = std::ceil(data_counts / (floor)(c2));

	for (int batch_id = 0; batch_id < batchCount; ++batch_id) {
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		caffe_test_net.Forward(dummy_blob_input_vec);
    boost::shared_ptr<Blob<float> > output = caffe_test_net.blob_by_name("fc8_seg");
		int bsize = output->num();

		for (int i = 0; i < bsize && counts < data_counts; i++, counts++) {
      Mat seg(DIM, DIM, CV_32FC1);
			for(int j = 0; j < DIM; j++) {
        for (int k = 0; k < DIM; k++) {
  				seg.at<float>(Point2d(k, j)) = (float) output->data_at(i, j * DIM + k, 0, 0);
        }
      }
      string fpath, tmp;
      float xmin, xmax, ymin, ymax;
      Mat finalS;
      fin >> fpath; getline(fin, tmp); // just eat up the remaining line
      fin_bbox >> tmp >> xmin >> ymin >> xmax >> ymax;
      genSegImg(xmin, ymin, xmax, ymax, seg, finalS);

      string outfpath = (fs::path(rootfolder) / fs::path(fpath)).string();
      makeParentDir(outfpath);
      if (METHOD == STOR_HDF5) {
        outfpath += ".h5";
        hid_t file_id_ = H5Fcreate(outfpath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_GE(file_id_, 0) << "Failed to open hdf5 file to store " << outfpath;
        hsize_t dims[2];
        dims[0] = finalS.rows; dims[1] = finalS.cols;
        H5LTmake_dataset_float(file_id_, "seg", 2, dims, (float*)finalS.data);
        herr_t st = H5Fclose(file_id_);
        CHECK_GE(st, 0) << "Unable to close h5 file for " << outfpath;
      } else if (METHOD == STOR_JPG) {
        Mat finalS_uint;
        finalS.convertTo(finalS_uint, CV_8UC1);
        equalizeHist(finalS_uint, finalS_uint);
        imwrite(outfpath, finalS_uint);
      }
		}
	}

	fin.close();
  fin_bbox.close();
	return 0;
}

void genSegImg(float xmin, float ymin, float xmax, float ymax, const Mat& S, Mat& res) {
  res = Mat(NW_IMG_HT, NW_IMG_WID, CV_32FC1);
  res.setTo(0);
  xmin = std::min(std::max(xmin + OFFSET, (float) 0), (float) NW_IMG_WID-1);
  ymin = std::min(std::max(ymin + OFFSET, (float) 0), (float) NW_IMG_HT-1);
  xmax = std::min(std::max(xmax + OFFSET, (float) 0), (float) NW_IMG_WID-1);
  ymax = std::min(std::max(ymax + OFFSET, (float) 0), (float) NW_IMG_HT-1);
  int x1 = xmin;
  int y1 = ymin;
  int x2 = xmax;
  int y2 = ymax;
  int height = MAX(1, y2 - y1 + 1);
  int width = MAX(1, x2 - x1 + 1);
  Mat extractedImage = res(Rect(x1, y1, width, height));
  Mat S2;
  resize(S, S2, Size(width, height));
  S2.copyTo(extractedImage);
}

