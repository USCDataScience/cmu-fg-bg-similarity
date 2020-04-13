/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "caffe/caffe.hpp"
#include "utils.hpp"
//#include "external/DiskVector/DiskVector.hpp"
#include "external/DiskVector/DiskVectorLMDB.hpp"
#include "lock.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

#define MAXFEATPERIMG 10000

void dumpFeature(FILE*, const vector<float>&);
long long hashCompleteName(long long, int);

int
main(int argc, char *argv[]) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "Extracting Features in CPU mode";
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  Caffe::set_phase(Caffe::TEST); // important, else will give random features

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("network-path,n", po::value<string>()->required(),
     "Path to the prototxt file")
    ("model-path,m", po::value<string>()->required(),
     "Path to corresponding caffemodel")
    ("outdir,o", po::value<string>()->default_value("output"),
     "Output directory")
    ("layer,l", po::value<string>()->default_value("pool5"),
     "CNN layer to extract features from")
    ("imgsdir,i", po::value<string>()->required(),
     "Input directory of images")
    ("imgslst,q", po::value<string>()->required(),
     "List of images relative to input directory")
    ("windir,w", po::value<string>()->default_value(""),
     "Input directory of all windows in each image (selective search format: y1 x1 y2 x2). Defaults to full image features.")
    ("output-type,t", po::value<string>()->default_value("lmdb"),
     "Output format [txt/lmdb]")
    ("normalize,y", po::bool_switch()->default_value(false),
     "Enable feature L2 normalization")
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

  fs::path NETWORK_PATH = fs::path(vm["network-path"].as<string>());
  fs::path MODEL_PATH = 
    fs::path(vm["model-path"].as<string>());
  string LAYER = vm["layer"].as<string>();
  fs::path OUTDIR = fs::path(vm["outdir"].as<string>());
  fs::path IMGSDIR = fs::path(vm["imgsdir"].as<string>());
  fs::path IMGSLST = fs::path(vm["imgslst"].as<string>());
  fs::path WINDIR = fs::path(vm["windir"].as<string>());
  string OUTTYPE = vm["output-type"].as<string>();
  bool NORMALIZE = vm["normalize"].as<bool>();

  NetParameter test_net_params;
  ReadProtoFromTextFile(NETWORK_PATH.string(), &test_net_params);
  Net<float> caffe_test_net(test_net_params);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(MODEL_PATH.string(), &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
  int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();

  // Get list of images in directory
  vector<fs::path> imgs;
  readList<fs::path>(IMGSLST, imgs);
  
  std::shared_ptr<DiskVectorLMDB<vector<float>>> dv;
  if (OUTTYPE.compare("lmdb") == 0) {
    dv = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
        new DiskVectorLMDB<vector<float>>(OUTDIR));
  }
  // Create output directory
  for (long long imgid = 1; imgid <= imgs.size(); imgid++) {
    fs::path imgpath = imgs[imgid - 1];

    LOG(INFO) << "Doing for " << imgpath << "...";

    vector<Mat> Is;
    Mat I = imread((IMGSDIR / imgpath).string());
    if (!I.data) {
      LOG(ERROR) << "Unable to read " << imgpath;
      continue;
    }
    vector<Rect> bboxes;
    if (WINDIR.string().size() > 0) {
      readBBoxesSelSearch<float>((WINDIR / (to_string((long long)imgid) + ".txt")).string(), bboxes);
    } else {
      bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
    }
    if (!I.data) {
      LOG(ERROR) << "Unable to read image " << imgpath;
    }
    // push in all subwindows
    for (int i = 0; i < bboxes.size(); i++) {
      Mat Itemp  = I(bboxes[i]);
      resize(Itemp, Itemp, Size(256, 256));
      Is.push_back(Itemp);
    }
    vector<vector<float>> output;
    /**
     * Separately computing features for either case of text/lmdb because 
     * can using locking (and run parallel) for text output
     */
    if (OUTTYPE.compare("text") == 0) {
      fs::path outFile = fs::change_extension(OUTDIR / imgpath, ".txt");
      if (!lock(outFile)) {
        continue;
      }
      computeFeatures<float>(caffe_test_net, Is, LAYER, BATCH_SIZE, output);
      if (NORMALIZE) {
        l2NormalizeFeatures(output);
      }
      fs::create_directories(outFile.parent_path());
      FILE* fout = fopen(outFile.string().c_str(), "w");
      for (int i = 0; i < output.size(); i++) {
        dumpFeature(fout, output[i]);
      }
      unlock(outFile);
      fclose(fout);
    } else if (OUTTYPE.compare("lmdb") == 0) {
      computeFeatures<float>(caffe_test_net, Is, LAYER, BATCH_SIZE, output);
      if (NORMALIZE) {
        l2NormalizeFeatures(output);
      }
      // output into a DiskVector
      for (int i = 0; i < output.size(); i++) {
        dv->Put(hashCompleteName(imgid, i), output[i]);
      }
    } else {
      LOG(ERROR) << "Unrecognized output type " << OUTTYPE << endl;
    }
  }

  return 0;
}

inline void dumpFeature(FILE* fout, const vector<float>& feat) {
  for (int i = 0; i < feat.size(); i++) {
    if (feat[i] == 0) {
      fprintf(fout, "0 ");
    } else {
      fprintf(fout, "%f ", feat[i]);
    }
  }
  fprintf(fout, "\n");
}

inline long long hashCompleteName(long long imgid, int id) {
  return (imgid - 1) * MAXFEATPERIMG + id;
}

