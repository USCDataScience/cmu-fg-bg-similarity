/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include <boost/archive/binary_oarchive.hpp>
#include <errno.h>
#include <curl/curl.h>
#include "caffe/caffe.hpp"
#include "utils.hpp"
// from the search code
#include "LSH.hpp"
#include "Resorter.hpp"
// for server
#include <zmq.h>

#define MAXFEATPERIMG 10000
#define TMP_PATH "./temp-dir/temp-img.jpg"

using namespace std;
using namespace std::chrono;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

Mat readFromURL(const string&);
string convertToFname(long long idx, const vector<fs::path>& imgslist, int);
string convertToFname_DEPRECATED(long long idx, const vector<fs::path>& imgslist, int);
void runSegmentationCode();
string getPartsFromFilePath(const fs::path& fpath, int nparts);

int
main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "Extracting Features in CPU mode";
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("network-path,n", po::value<string>()->required(),
     "Path to the prototxt file")
    ("model-path,m", po::value<string>()->required(),
     "Path to corresponding caffemodel")
    ("layer,l", po::value<string>()->default_value("pool5"),
     "CNN layer to extract features from")
    ("index,i", po::value<string>()->required(),
     "Path to load search index from")
    ("featstor,s", po::value<string>()->required(),
     "Path to feature store")
    ("imgslist,q", po::value<string>()->required(),
     "File with images list")
    ("port-num,p", po::value<string>()->default_value("5555"),
     "Port to run the service on")
    ("seg-img,g", po::value<string>()->default_value(""),
     "Path to read the segmentation image from. Keep empty for full image search. "
     "On setting this, system will pool features from bg boxes")
    ("deprecated-model,d", po::bool_switch()->default_value(false),
     "Set if using a deprecated model, which has images indexed from "
     "0. In future, all image ids, and box ids are 1 indexed")
    ("compressedFeatStor", po::bool_switch()->default_value(false),
     "Set if using a compressed feature store. This can't currently "
     "read this information from the lmdb")
    ("duplist", po::value<fs::path>()->default_value(""),
     "Path to list with unique/duplicate entried. Will augment the "
     "output with duplicate images")
    ("num-output", po::value<int>()->default_value(100),
     "Max number of matches to return")
    ("nRerank", po::value<int>()->default_value(5000),
     "Max number of images to re-rank using actual features. "
     "Determines the test time performance vs speed tradeoff. "
     "Large n implies better results but slower performance.")
    ("nPathParts", po::value<int>()->default_value(1),
     "Number of parts of the image fpath (from image file list) to print out")
    ("foreground,f", po::value<bool>()->default_value(0),
     "Extract the features of the foreground")
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
 
  bool FOREGROUND = vm["foreground"].as<bool>();
  cout << "FOREGROUND: " << FOREGROUND << endl;
  int nPathParts = vm["nPathParts"].as<int>();
  LOG(INFO) << "Printing " << nPathParts << " of filenames to output";
  fs::path NETWORK_PATH = fs::path(vm["network-path"].as<string>());
  fs::path MODEL_PATH = 
    fs::path(vm["model-path"].as<string>());
  string LAYER = vm["layer"].as<string>();
  fs::path SEG_IMG_PATH = fs::path(vm["seg-img"].as<string>());
  int nRerank = vm["nRerank"].as<int>();
  bool DEPRECATED_MODEL = vm["deprecated-model"].as<bool>();
  vector<string> layers = {LAYER};
  vector<fs::path> imgslist;
  CNNFeatureUtils::readList_withSpaces(vm["imgslist"].as<string>(), imgslist);
  int num_output = vm["num-output"].as<int>();

  Net<float> caffe_test_net(NETWORK_PATH.string(), caffe::TEST);
  caffe_test_net.CopyTrainedLayersFrom(MODEL_PATH.string());
  int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();
  
  // Read the search index
  LOG(INFO) << "Reading the search index...";
  ifstream ifs(vm["index"].as<string>(), ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  std::shared_ptr<LSH> l(new LSH(0,0));
  ia >> *l;
  ifs.close();
  LOG(INFO) << "Done.";

  LOG(INFO) << "Setting up the server...";
  auto featstor = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
      new DiskVectorLMDB<vector<float>>(vm["featstor"].as<string>(), 1, vm["compressedFeatStor"].as<bool>()));

  //  Socket to talk to clients
  void *context = zmq_ctx_new();
  void *responder = zmq_socket(context, ZMQ_REP);
  int rc = zmq_bind(responder, (string("tcp://*:")
        + vm["port-num"].as<string>()).c_str());
  assert (rc == 0);

  LOG(INFO) << "Server Ready";

  while (true) {
    char buffer[1000], outbuf[1000];
    ostringstream oss;
    zmq_recv (responder, buffer, 1000, 0);
    LOG(INFO) << "Recieved: " << buffer;
    high_resolution_clock::time_point st = high_resolution_clock::now();

    vector<Mat> Is;
    Mat I = readFromURL(string(buffer));
    if (!I.data) {
      LOG(ERROR) << "Unable to read " << buffer;
      oss << "Unable to read " << buffer;
      zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
      continue;
    }

    vector<Rect> bboxes;
    if (SEG_IMG_PATH.string().length() > 0) {
      if (! imwrite(TMP_PATH, I)) {
        oss << "Unable to write query image to " << TMP_PATH 
             << " to run segmentation service";
        cerr << oss.str() << endl;
        zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
        continue;
      }
      runSegmentationCode();
      Mat S; // not really used
      CNNFeatureUtils::genSlidingWindows(I.size(), bboxes);
      CNNFeatureUtils::pruneBboxesWithSeg(I.size(), SEG_IMG_PATH, bboxes, S, FOREGROUND);
    } else {
      bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
    }

    for (int i = 0; i < bboxes.size(); i++) {
      Mat Itemp = I(bboxes[i]);
      resize(Itemp, Itemp, Size(256, 256));
      Is.push_back(Itemp);
    }

    high_resolution_clock::time_point read = high_resolution_clock::now();

    vector<vector<vector<float>>> feats;
    CNNFeatureUtils::computeFeaturesPipeline<float>(caffe_test_net, Is, 
        layers, BATCH_SIZE, feats, true, "avg", true);

    high_resolution_clock::time_point feat = high_resolution_clock::now();

    unordered_set<long long int> init_matches;
    vector<pair<float, long long int>> res;
    l->search(feats[0][0], init_matches, nRerank);
    LOG(INFO) << "Re-sorting " << init_matches.size() << " matches";
    Resorter::resort_multicore(init_matches, featstor, feats[0][0], res);
    if (vm["duplist"].as<fs::path>().string().length() > 0) {
      res = augmentWithDuplicates(vm["duplist"].as<fs::path>(), res);
      LOG(INFO) << "Augmented to get " << res.size() << " matches";
    }

    high_resolution_clock::time_point search = high_resolution_clock::now();

    for (int i = 0; i < min(res.size(), (size_t) num_output); i++) {
      oss << res[i].first << ":" 
          << (DEPRECATED_MODEL ? convertToFname_DEPRECATED(res[i].second, imgslist, nPathParts)
                               : convertToFname(res[i].second, imgslist, nPathParts)) 
          << ',';
    }
    zmq_send(responder, oss.str().c_str(), oss.str().length(), 0);
    LOG(INFO) << "Time taken: " << endl
              << " Read : " << duration_cast<milliseconds>(read - st).count() << "ms" << endl
              << " Ext Feat : " << duration_cast<milliseconds>(feat - read).count() << "ms" << endl
              << " Search : " << duration_cast<milliseconds>(search - feat).count() << "ms" << endl;
  }
  return 0;
}

/*
int readFromURL(const string& url, Mat& I) {
  string temppath = TMP_PATH;
  int ret = system((string("wget --no-check-certificate ") + url + " -O " + temppath).c_str());
  I = imread(temppath.c_str());
  return ret;
}
*/

//curl writefunction to be passed as a parameter
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata) {
    std::ostringstream *stream = (std::ostringstream*)userdata;
    size_t count = size * nmemb;
    stream->write(ptr, count);
    return count;
}

//function to retrieve the image as Cv::Mat data type
Mat readFromURL(const string& url) {
  CURL *curl;
  CURLcode res;
  std::ostringstream stream;
  curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str()); //the img url
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L); // don't verify
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // don't verify
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // follow re-directs
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr when the writefunction is called
  res = curl_easy_perform(curl); // start curl
  std::string output = stream.str(); // convert the stream into a string
  curl_easy_cleanup(curl); // cleanup
  std::vector<char> data = std::vector<char>( output.begin(), output.end() ); //convert string into a vector
  cv::Mat data_mat = cv::Mat(data); // create the cv::Mat datatype from the vector
  cv::Mat image = cv::imdecode(data_mat,1); //read an image from memory buffer
  return image;
}

string convertToFname(long long idx, const vector<fs::path>& imgslist, int nparts) {
  size_t txid = idx / MAXFEATPERIMG - 1;
  CHECK_GT(imgslist.size(), txid) << "File doesn't have enough lines";
  return getPartsFromFilePath(imgslist[txid], nparts);
}

string convertToFname_DEPRECATED(long long idx, const vector<fs::path>& imgslist, int nparts) {
  size_t txid = idx / MAXFEATPERIMG;
  CHECK_GT(imgslist.size(), txid) << "File doesn't have enough lines";
  return getPartsFromFilePath(imgslist[txid], nparts);
}

string getPartsFromFilePath(const fs::path& fpath, int nparts) {
  if (nparts == 1) {
    return fpath.filename().string();
  } else if (nparts == -1) {
    // return the whole thing
    return fpath.string();
  } else {
    LOG(FATAL) << "nparts = " << nparts << " is not implemented.";
  }
}

void runSegmentationCode() {
  static bool initializedSocket = false;
  static void *context = NULL;
  static void *socket = NULL;
  if (!initializedSocket) {
    context = zmq_ctx_new();
    socket = zmq_socket(context, ZMQ_REQ);
    if (zmq_connect(socket, "tcp://localhost:5559") == -1) {
      LOG(ERROR) << "Unable to connect to segmentation service. " << strerror(errno);
    }
    initializedSocket = true;
  }
  string path_to_send = boost::filesystem::canonical(fs::path(TMP_PATH)).string() + "\0";
  cout << "sedning : " << path_to_send;
  zmq_send(socket, path_to_send.c_str(), path_to_send.length(), 0);
  char temp[1000];
  zmq_recv(socket, temp, 1000, 0); // wait till answer received
  LOG(ERROR) << temp;
}

