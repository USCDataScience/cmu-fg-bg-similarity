#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "Config.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace fs = boost::filesystem;

namespace CNNFeatureUtils {

template<typename Dtype>
void computeFeatures(Net<Dtype>& caffe_test_net,
    const vector<Mat>& imgs,
    const vector<string>& LAYERS,
    int BATCH_SIZE,
    vector<vector<vector<Dtype>>>& output,
    bool verbose = true) {
  output.clear();
  // initialize output for each layers
  for (int i = 0; i < LAYERS.size(); i++) {
    output.push_back(vector<vector<Dtype>>());
  }
  int nImgs = imgs.size();
  int nBatches = ceil(nImgs * 1.0f / BATCH_SIZE);
  for (int batch = 0; batch < nBatches; batch++) {
    int actBatchSize = min(nImgs - batch * BATCH_SIZE, BATCH_SIZE);
    vector<Mat> imgs_b;
    if (actBatchSize >= BATCH_SIZE) {
      imgs_b.insert(imgs_b.end(), imgs.begin() + batch * BATCH_SIZE, 
          imgs.begin() + (batch + 1) * BATCH_SIZE);
    } else {
      imgs_b.insert(imgs_b.end(), imgs.begin() + batch * BATCH_SIZE, imgs.end());
      for (int j = actBatchSize; j < BATCH_SIZE; j++)
        imgs_b.push_back(imgs[0]);
    }
    vector<int> dvl(BATCH_SIZE, 0);
//    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype>>(
//        caffe_test_net.layers()[0])->AddMatVector(imgs_b, dvl);
    boost::shared_ptr<MemoryDataLayer<float>> md_layer =
          boost::dynamic_pointer_cast<MemoryDataLayer<float>>(caffe_test_net.layers()[0]);
    md_layer->AddMatVector(imgs_b, dvl);
    Dtype loss = 0.0f;
    caffe_test_net.ForwardPrefilled(&loss);
    for (int l = 0; l < LAYERS.size(); l++) {
      const boost::shared_ptr<Blob<Dtype>> feat = 
        caffe_test_net.blob_by_name(LAYERS[l]);
      for (int i = 0; i < actBatchSize; i++) {
        Dtype* feat_data = feat->mutable_cpu_data() + feat->offset(i);
        output[l].push_back(vector<Dtype>(feat_data, feat_data + feat->count() / feat->shape()[0]));
      }
    }
    if (verbose) {
      LOG(INFO) << "Batch " << batch << "/" << nBatches << " (" << actBatchSize << " images) done";
    }
  }
}

/**
 * Function to return list of images in a directory (searched recursively).
 * The output paths are w.r.t. the path imgsDir
 */
void genImgsList(const fs::path& imgsDir, vector<fs::path>& list) {
  if(!fs::exists(imgsDir) || !fs::is_directory(imgsDir)) return;
  vector<string> imgsExts = {".jpg", ".png", ".jpeg", ".JPEG", ".PNG", ".JPG"};

  fs::recursive_directory_iterator it(imgsDir);
  fs::recursive_directory_iterator endit;
  while(it != endit) {
    if(fs::is_regular_file(*it) && 
        find(imgsExts.begin(), imgsExts.end(), 
          it->path().extension()) != imgsExts.end())
      // write out paths but clip out the initial relative path from current dir 
      list.push_back(fs::path(it->path().relative_path().string().
            substr(imgsDir.relative_path().string().length())));
    ++it;
  }
  LOG(INFO) << "Found " << list.size() << " image file(s) in " << imgsDir;
}

/**
 * Read bbox from file in selsearch format (y1 x1 y2 x2)
 */
template<typename Dtype>
void readBBoxesSelSearch(const fs::path& fpath, vector<Rect>& output) {
  Dtype x1, x2, y1, y2;
  output.clear();
  ifstream ifs(fpath.string());
  if (!ifs.is_open()) {
    LOG(ERROR) << "Unable to open file " << fpath.string();
    return;
  }
  string line;
  while (getline(ifs, line)) {
    replace(line.begin(), line.end(), ',', ' ');
    istringstream iss(line);
    iss >> y1 >> x1 >> y2 >> x2;
    output.push_back(Rect(x1 - 1, y1 - 1, x2 - x1, y2 - y1)); 
  }
  ifs.close();
}

/**
 * Read any list of Dtype element separated by a whitespace delimitter.
 */
template<typename Dtype>
void readList(const fs::path& fpath, vector<Dtype>& output) {
  output.clear();
  Dtype el;
  ifstream ifs(fpath.string());
  if (!ifs.is_open()) {
    LOG(FATAL) << "Unable to open file " << fpath;
  }

  cout << "ReadList Function" << endl;

  while (ifs >> el) {
    output.push_back(el);
  }
  ifs.close();
}

/**
 * Read any list of Dtype element with spaces, separated by newline
 */
template<typename Dtype>
void readList_withSpaces(const fs::path& fpath, vector<Dtype>& output) {
  output.clear();
  string el_str;
  ifstream ifs(fpath.string());
  if (!ifs.is_open()) {
    LOG(FATAL) << "Unable to open file " << fpath;
  }
  while (getline(ifs, el_str)) {
    output.push_back(Dtype(el_str));
  }
  ifs.close();
}

template<typename Dtype>
void l2NormalizeFeatures(vector<vector<Dtype>>& feats) {
  #pragma omp parallel for
  for (int i = 0; i < feats.size(); i++) {
    Dtype l2norm = 0;
    for (auto el = feats[i].begin(); el != feats[i].end(); el++) {
      l2norm += (*el) * (*el);
    }
    l2norm = sqrt(l2norm);
    for (int j = 0; j < feats[i].size(); j++) {
      feats[i][j] = feats[i][j] / l2norm;
    }
  } 
}

void genSlidingWindows(const Size& I_size, vector<Rect>& bboxes) {
  bboxes.clear();
  int sliding_sz_x = max((int) (SLIDINGWIN_WINDOW_RATIO * I_size.width),
      SLIDINGWIN_MIN_SZ_X);
  int sliding_sz_y = max((int) (SLIDINGWIN_WINDOW_RATIO * I_size.height),
      SLIDINGWIN_MIN_SZ_Y);
  sliding_sz_x = sliding_sz_y = min(sliding_sz_x, sliding_sz_y);
  int sliding_stride_x = max((int) (SLIDINGWIN_STRIDE_RATIO * sliding_sz_x),
      SLIDINGWIN_MIN_STRIDE);
  int sliding_stride_y = max((int) (SLIDINGWIN_STRIDE_RATIO * sliding_sz_y),
      SLIDINGWIN_MIN_STRIDE);
  sliding_stride_x = sliding_stride_y = min(sliding_stride_x, sliding_stride_y);
  for (int x = 0; x < I_size.width - sliding_sz_x; x += sliding_stride_x) {
    for (int y = 0; y < I_size.height - sliding_sz_y; y += sliding_stride_y) {
      bboxes.push_back(Rect(x, y, sliding_sz_x, sliding_sz_y));
    }
  }
}

void pruneBboxesWithSeg(const Size& I_size, 
    const fs::path& segpath, vector<Rect>& bboxes, Mat& S, bool FOREGROUND) {
  // TODO (rg): speed up by using integral images
  //cout << "DEBUG: pruneBboxes 1" << endl;
  vector<Rect> res;
  vector<float> overlaps;
  //cout << segpath.string() << endl;
  S = imread(segpath.string().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
  //cout << "DEBUG: imread finished" << endl;
  // resize to the same size as I
  resize(S, S, I_size);
  //cout << "DEBUG: resize finished" << endl;
  //cout << "DEBUG: pruneBboxes 2" << endl;
  for (int i = 0; i < bboxes.size(); i++) {
    int in = cv::sum(S(bboxes[i]))[0]; 
    int tot = bboxes[i].width * bboxes[i].height;
    float ov = in * 1.0f / tot;
    overlaps.push_back(ov);
    if (!FOREGROUND) {
        if (in * 1.0f / tot < PERC_FGOVERLAP_FOR_BG) { // bg patch
            res.push_back(bboxes[i]);
        }
    } else {
        if (in * 1.0f / tot >= PERC_FGOVERLAP_FOR_BG) { // fg patch
            res.push_back(bboxes[i]);
        } 
    }
  }
  if (res.size() == 0) {
    LOG(ERROR) << "pruneBboxesWithSeg: No patches qualified for background.";
    if (bboxes.size() > 0) {
      int min_pos = distance(overlaps.begin(), 
          min_element(overlaps.begin(), overlaps.end()));
      res.push_back(bboxes[min_pos]);
      LOG(ERROR) << "pruneBboxesWithSeg: Pushing in the min-overlap box (dist= "
                 << overlaps[min_pos] << " at " << min_pos << ")";
    } else {
      LOG(ERROR) << "pruneBboxesWithSeg: There are no boxes, returning the full image";
      res.push_back(Rect(0, 0, I_size.width, I_size.height));
    }
  }
  if (FOREGROUND && I_size.width > 10 && I_size.height > 10) {
      int minx, maxx, miny, maxy;
      minx = res[0].x;
      miny = res[0].y;
      maxx = minx + res[0].width - 1;
      maxy = maxy + res[0].height - 1;
      for (int i = 0; i< res.size(); i++) {
        minx = min(res[i].x, minx);
        miny = min(res[i].y, miny);
        maxx = max(res[i].x + res[i].width - 1, maxx);
        maxy = max(res[i].y + res[i].height - 1, maxy);
      }
      res.clear();
      res.push_back(Rect(minx, miny, maxx - minx, maxy - miny));
  }

  bboxes = res;
}

void pruneBboxesWithSeg(const Size& I_size, 
    const Mat& S_orig, vector<Rect>& bboxes, bool FOREGROUND) {
  // TODO (rg): speed up by using integral images
  vector<Rect> res;
  // resize to the same size as I
  Mat S;
  resize(S_orig, S, I_size);
  for (int i = 0; i < bboxes.size(); i++) {
    int in = cv::sum(S(bboxes[i]))[0]; 
    int tot = bboxes[i].width * bboxes[i].height;
    if (!FOREGROUND) {
        if (in * 1.0f / tot < PERC_FGOVERLAP_FOR_BG) { // bg patch
            res.push_back(bboxes[i]);
        }
    } else {
        if (in * 1.0f / tot >= PERC_FGOVERLAP_FOR_BG) { // fg patch
            res.push_back(bboxes[i]);
        }
    }
  }
  bboxes = res;
}


void DEBUG_storeWindows(const vector<Mat>& Is, fs::path fpath, 
    const Mat& I, const Mat& S) {
  fs::create_directories(fpath);
  imwrite(fpath.string() + "/main.jpg", I);
  imwrite(fpath.string() + "/seg.jpg", S);
  for (int i = 0; i < Is.size(); i++) {
    imwrite(fpath.string() + "/" + to_string((long long)i) + ".jpg", Is[i]);
  }
}

void poolFeatures(vector<vector<float>>& feats, const string& pooltype) {
  if (feats.size() == 0) return;
  vector<float> res(feats[0].size(), 0.0f);
  if (pooltype.compare("avg") == 0) {
    for (int i = 0; i < feats.size(); i++) {
      for (int j = 0; j < feats[i].size(); j++) {
        res[j] += feats[i][j];
      }
    }
    for (int j = 0; j < res.size(); j++) {
      res[j] /= feats.size();
    }
  } else {
    LOG(ERROR) << "Pool type " << pooltype << " not implemented yet!";
  }
  feats.clear();
  feats.push_back(res);
}

template <typename Dtype>
void convertBlobToMat(const Blob<Dtype>& blob, Mat& mat, int n = 0) {
  // n defines the specific image in the blob
  int wd = blob.width();
  int ht = blob.height();
  int ch = blob.channels();
  mat = Mat(ht, wd, CV_8UC3);
  for (int c = 0; c < ch; c++) {
    for (int h = 0; h < ht; h++) {
      for (int w = 0; w < wd; w++) {
        mat.at<Vec3b>(h, w)[c] = (uint8_t) blob.data_at(n, c, h, w);
      }
    }
  }
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

void getUniqueIds(const fs::path& fpath, vector<long long>& ids, long long start_img_id = 1) {
  ids.clear();
  ifstream fin(fpath.string().c_str());
  string line;
  long long lno = 0;
  while (getline(fin, line)) {
    lno += 1;
    if (lno < start_img_id) {
      continue;
    }
    if (line[0] == 'U') {
      ids.push_back(lno);
    }
  }
  fin.close();
}

}

#endif

