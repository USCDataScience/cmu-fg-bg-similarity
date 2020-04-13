/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "caffe/caffe.hpp"
#include "utils.hpp"

using namespace std;
using namespace std::chrono;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

template<typename Dtype>
void computeFeaturesPipeline(Net<Dtype>& caffe_test_net,
    const vector<Mat>& Is,
    const vector<string>& layers,
    int BATCH_SIZE,
    vector<vector<vector<Dtype>>>& output,
    bool verbose,
    const string& POOLTYPE,
    bool NORMALIZE);
void genSegImg(float xmin, float ymin, float xmax, float ymax, const Mat& S, Mat& res, int, int);

int
main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "Extracting Features in CPU mode";
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("loc-network-path,n", po::value<string>()->required(),
     "Path to the localization prototxt file")
    ("loc-model-path,m", po::value<string>()->required(),
     "Path to localization caffemodel")
    ("seg-network-path,p", po::value<string>()->required(),
     "Path to the segmentation prototxt file")
    ("seg-model-path,q", po::value<string>()->required(),
     "Path to segmentation caffemodel")
    ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    LOG(INFO) << desc;
    return -1;
  }
  try {
    po::notify(vm);
  } catch(po::error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  fs::path LOC_NETWORK_PATH = fs::path(vm["loc-network-path"].as<string>());
  fs::path LOC_MODEL_PATH = 
    fs::path(vm["loc-model-path"].as<string>());
  fs::path SEG_NETWORK_PATH = fs::path(vm["seg-network-path"].as<string>());
  fs::path SEG_MODEL_PATH = 
    fs::path(vm["seg-model-path"].as<string>());

  Net<float> loc_caffe_test_net(LOC_NETWORK_PATH.string(), caffe::TEST);
  loc_caffe_test_net.CopyTrainedLayersFrom(LOC_MODEL_PATH.string());
  int LOC_BATCH_SIZE = loc_caffe_test_net.blob_by_name("data")->num();
  vector<string> loc_layers = {"fc8_loc"};
  Net<float> seg_caffe_test_net(SEG_NETWORK_PATH.string(), caffe::TEST);
  seg_caffe_test_net.CopyTrainedLayersFrom(SEG_MODEL_PATH.string());
  int SEG_BATCH_SIZE = seg_caffe_test_net.blob_by_name("data")->num();
  vector<string> seg_layers = {"fc8_seg"};

  string imgpath = "/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/AbuSimbel/people_1.jpg";
  //string imgpath = "/home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Images/corpus/ImagesTexas/Texas_2012_10_10_1349841918000_4_0.jpg";
  //string imgpath = "/home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Images/corpus/ImagesTexas/Texas_2012_10_10_1349846732000_6_5.jpg";
  //string imgpath = "/home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Images/corpus/ImagesCalifornia/California_image_2012_7_9_1341859695000_2_0.jpg";
  vector<Mat> Is;
  Mat I = imread(imgpath);
  if (!I.data) {
    LOG(ERROR) << "Unable to read " << imgpath;
    return -1;
  }
  resize(I, I, Size(256, 256));
  Is.push_back(I);
  // [layer[image[feature]]]
  vector<vector<vector<float>>> loc_output;
  computeFeaturesPipeline(loc_caffe_test_net, Is, loc_layers, 
      LOC_BATCH_SIZE, loc_output, /* verbose= */ false, 
      /* POOLTYPE= */ "", /* NORMALIZE= */ false);
 
  float OFFSET = (256 - 227) / 2.0;
  /*
  loc_output[0][0][0] += OFFSET;
  loc_output[0][0][1] += OFFSET;
  loc_output[0][0][2] += OFFSET;
  loc_output[0][0][3] += OFFSET;
  */
   
  cout << loc_output[0][0][0] << " "
       << loc_output[0][0][1] << " "
       << loc_output[0][0][2] << " "
       << loc_output[0][0][3];
  
  int width = I.cols;
  int height = I.rows;
  float xmin = std::min(width - 1.0f, std::max(0.0f, loc_output[0][0][0] + OFFSET));
  float ymin = std::min(height - 1.0f, std::max(0.0f, loc_output[0][0][1] + OFFSET));
  float xmax = std::min(width - 1.0f, std::max(0.0f, loc_output[0][0][2] + OFFSET));
  float ymax = std::min(height - 1.0f, std::max(0.0f, loc_output[0][0][3] + OFFSET));

  Rect roi(xmin,
           ymin,
           xmax - xmin,
           ymax - ymin);
  Mat T = I(roi);
  resize(T, T, Size(55, 55));

  caffe::BlobProto mean_blob_proto;
  Mat mean;
  caffe::ReadProtoFromBinaryFile("/home/rgirdhar/data/Work/Code/0005_ObjSegment/nips14_loc_seg_testonly/Caffe_Segmentation/segscripts/models/mean.binaryproto", &mean_blob_proto);
  caffe::Blob<unsigned int> mean_blob;
  mean_blob.FromProto(mean_blob_proto);
  convertBlobToMat(mean_blob, mean);
  mean = mean(roi);
  resize(mean, mean, Size(55, 55));
  /*
  double minEl, maxEl;
  minMaxLoc(mean, &minEl, &maxEl);
  cout << " mx  " << minEl << " " << maxEl << endl;
  minMaxLoc(T, &minEl, &maxEl);
  cout << " mx  " << minEl << " " << maxEl << endl;
  */
  
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 55; h++) {
      for (int w = 0; w < 55; w++) {
        T.at<Vec3b>(h, w)[c] = (uint8_t) T.at<Vec3b>(h, w)[c] - mean.at<Vec3b>(h, w)[c];
      }
    }
  }
  

  //T = T - mean;
  flip(T, T, 1);
//  cout << T;
  imwrite("mean.jpg", T);

  vector<Mat> Ts;
  Ts.push_back(T);
  vector<vector<vector<float>>> seg_output;
  computeFeaturesPipeline(seg_caffe_test_net, Ts, seg_layers, 
      SEG_BATCH_SIZE, seg_output, /* verbose= */ false, 
      /* POOLTYPE= */ "", /* NORMALIZE= */ false);
  Mat Res(50, 50, CV_32FC1);
  for (int i = 0; i < 50; i++) {
    for (int j = 0; j < 50; j++) {
      Res.at<float>(i, j) = seg_output[0][0][i * 50 + j];
    }
  }
//  flip(Res, Res, 1);
  Mat seg;
  genSegImg(xmin, ymin, xmax, ymax, Res, seg, I.rows, I.cols);
  Mat seg_uint;
  seg.convertTo(seg_uint, CV_8UC1);
  equalizeHist(seg_uint, seg_uint);
  //resize(seg_uint, seg_uint, I.size());
  imwrite("final.jpg", seg_uint);
  vector<Mat> channels(3);
  split(I, channels);
  divide(seg_uint, Scalar(255), seg_uint);
  multiply(seg_uint, channels[0], channels[0]);
  multiply(seg_uint, channels[1], channels[1]);
  multiply(seg_uint, channels[2], channels[2]);
  merge(channels, I);
  // normalize(Res, Res, 0, 255, NORM_MINMAX, CV_8UC1);
  
  // equalizeHist(Res, Res);
  imwrite("over.jpg", I);
  return 0;
}

template<typename Dtype>
void computeFeaturesPipeline(Net<Dtype>& caffe_test_net,
    const vector<Mat>& Is,
    const vector<string>& layers,
    int BATCH_SIZE,
    vector<vector<vector<Dtype>>>& output,
    bool verbose,
    const string& POOLTYPE,
    bool NORMALIZE) {
  computeFeatures(caffe_test_net, Is, layers, BATCH_SIZE, output, verbose);
  if (! POOLTYPE.empty()) {
    // assuming all layers need to be pooled
    for (int l = 0; l < output.size(); l++) {
      poolFeatures(output[l], POOLTYPE);
    }
  }
  if (NORMALIZE) {
    // assuming all layers need to be normalized
    for (int i = 0; i < output.size(); i++) {
      l2NormalizeFeatures(output[i]);
    }
  }
}

void genSegImg(float xmin, float ymin, float xmax, float ymax, const Mat& S, Mat& res,
    int NW_IMG_HT, int NW_IMG_WID) {
  res = Mat(NW_IMG_HT, NW_IMG_WID, CV_32FC1);
  res.setTo(0);
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

