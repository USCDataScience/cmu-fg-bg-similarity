/**
 * Code to compute CNN (ImageNet) features for a given Full image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "caffe/caffe.hpp"
#include "utils.hpp"
#include "external/DiskVector/DiskVectorLMDB.hpp"
#include "lock.hpp"

#define MAXFEATPERIMG 10000

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

void dumpFeature(FILE*, const vector<float>&);
long long hashCompleteName(long long, int);
void computeAndStore(const vector<Mat>& Is,
    const vector<long long>& imgids,
    const vector<fs::path>& imgpaths,
    Net<float>& caffe_test_net,
    const string& LAYER,
    const fs::path& OUTDIR,
    int BATCH_SIZE,
    std::shared_ptr<DiskVectorLMDB<vector<float>>> dv,
    const string& OUTTYPE,
    bool NORMALIZE);
long long getHash(long long id);


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
    ("output-type,t", po::value<string>()->default_value("lmdb"),
     "Output format [txt/lmdb]")
    ("normalize,y", po::bool_switch()->default_value(false),
     "Enable feature L2 normalization")
    ("start,s", po::value<long long>()->default_value(1),
     "Image index (row number - 1 indexed - in the ImgsList file) to start with")
    ("compressedFeatStor", po::bool_switch()->default_value(false),
     "Store features in LMDB with zlib compression")
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
  readList_withSpaces<fs::path>(IMGSLST, imgs);
  
  std::shared_ptr<DiskVectorLMDB<vector<float>>> dv;
  if (OUTTYPE.compare("lmdb") == 0) {
    dv = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
        new DiskVectorLMDB<vector<float>>(OUTDIR, 0, vm["compressedFeatStor"].as<bool>()));
  }
  // Create output directory
  vector<Mat> Is;
  vector<long long> imgids;
  vector<fs::path> imgpaths;
  for (long long imgid = vm["start"].as<long long>(); imgid <= imgs.size(); imgid++) {
    imgids.push_back(imgid);
    fs::path imgpath = imgs[imgid - 1];
    imgpaths.push_back(imgpath);

    Mat I = imread((IMGSDIR / imgpath).string());
    if (!I.data) {
      LOG(ERROR) << "Unable to read " << imgpath;
      continue;
    }
    resize(I, I, Size(256, 256));
    Is.push_back(I);
    if (Is.size() >= BATCH_SIZE) {
      computeAndStore(Is, imgids, imgpaths,
          caffe_test_net, LAYER, OUTDIR,
          BATCH_SIZE, dv, OUTTYPE, NORMALIZE);
      Is.clear();
      imgids.clear();
      imgpaths.clear();
      LOG(INFO) << "Done uptil " << imgid;
    }
  }
  computeAndStore(Is, imgids, imgpaths,
      caffe_test_net, LAYER, OUTDIR, 
      BATCH_SIZE, dv, OUTTYPE, NORMALIZE);
  LOG(INFO) << "Done All";
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

void computeAndStore(const vector<Mat>& Is,
    const vector<long long>& imgids,
    const vector<fs::path>& imgpaths,
    Net<float>& caffe_test_net,
    const string& LAYER,
    const fs::path& OUTDIR,
    int BATCH_SIZE,
    std::shared_ptr<DiskVectorLMDB<vector<float>>> dv,
    const string& OUTTYPE,
    bool NORMALIZE) {
  vector<vector<float>> output;
  computeFeatures(caffe_test_net, Is, LAYER, BATCH_SIZE, output);
  if (NORMALIZE) {
    l2NormalizeFeatures(output);
  }
  if (OUTTYPE.compare("text") == 0) {
    for (int i = 0; i < output.size(); i++) {
      fs::path outFile = fs::change_extension(OUTDIR / imgpaths[i], ".txt");
      fs::create_directories(outFile.parent_path());
      FILE* fout = fopen(outFile.string().c_str(), "w");
      dumpFeature(fout, output[i]);
      fclose(fout);
    }
  } else if (OUTTYPE.compare("lmdb") == 0) {
    // output into the DiskVector
    for (int i = 0; i < output.size(); i++) {
      dv->Put(getHash(imgids[i]), output[i]);
    }
  } else {
    LOG(ERROR) << "Unrecognized output type " << OUTTYPE << endl;
  }
}

long long getHash(long long id) { // id is 1 indexed
  // TODO FIX THIS. Use consistent
  // Since Saturday 11 April 2015 02:06:40 AM GMT : using both 1 indexed. 
  return id * MAXFEATPERIMG + 1;
}

