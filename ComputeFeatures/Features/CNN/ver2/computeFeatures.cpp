/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include <hdf5.h>
#include <zmq.hpp>
#include "caffe/caffe.hpp"
#include "utils.hpp"
//#include "external/DiskVector/DiskVector.hpp"
#include "external/DiskVector/DiskVectorLMDB.hpp"
#include "lock.hpp"

using namespace std;
using namespace std::chrono;
using namespace caffe;
using namespace cv;
using namespace CNNFeatureUtils;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

#define MAXFEATPERIMG 10000
#define PRINT_INTERVAL 20
// output type
#define OUTTYPE_TEXT 1
#define OUTTYPE_LMDB 2
#define OUTTYPE_HDF5 3

void dumpFeatures_txt(const fs::path&, const vector<vector<float>>&);
void dumpFeatures_hdf5(const fs::path&, vector<vector<float>>&);
long long hashCompleteName(long long, int);
void readImageUsingService(const string&, Mat&);
void readSegUsingService(const Mat&, Mat&);

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
     "Input directory of all windows in each image (selective search format: y1 x1 y2 x2)." 
     "Defaults to full image features."
     "Ignores sliding window if set.")
    ("sliding,s", po::value<bool>()->default_value(0),
     "Compute features in sliding window fashion")
    ("pool,p", po::value<string>()->default_value(""),
     "Pool the features from different regions into one feature."
     "Supports: <empty>: no pooling. store all features."
     "avg: avg pooling")
    ("segdir,x", po::value<string>()->default_value(""),
     "Directory with images with same filename as in the corpus images dir "
     "but uses it to prune the set of windows. "
     "By default keeps only those overlapping <0.2 with FG")
    ("seglist", po::value<string>()->default_value(""),
     "List with the paths (wrt segdir) for segmentations for each image. "
     "By default it uses from imgslst.")
    ("startimgid,z", po::value<long long>()->default_value(1),
     "The image id of the first image in the list."
     "Useful for testing parts of dataset because the selsearch boxes" 
     "etc use the image ids. Give 1 indexed")
    ("output-type,t", po::value<string>()->default_value("lmdb"),
     "Output format [txt/lmdb/hdf5]")
    ("normalize,y", po::value<bool>()->default_value(0),
     "Enable feature L2 normalization")
    ("ids2compute4", po::value<string>()->default_value(""),
     "File with list of image ids (1 indexed) to compute the features"
     "for. If not specified, computes from start img idx to end")
    ("uniquelist", po::value<string>()->default_value(""),
     "File with unique/duplicate information using SHA1")
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

  fs::path NETWORK_PATH = fs::path(vm["network-path"].as<string>());
  fs::path MODEL_PATH = 
    fs::path(vm["model-path"].as<string>());
  string LAYERS = vm["layer"].as<string>();
  vector<string> layers;
  boost::split(layers, LAYERS, boost::is_any_of(","));
  fs::path OUTDIR = fs::path(vm["outdir"].as<string>());
  fs::path IMGSDIR = fs::path(vm["imgsdir"].as<string>());
  fs::path IMGSLST = fs::path(vm["imgslst"].as<string>());
  fs::path WINDIR = fs::path(vm["windir"].as<string>());
  fs::path SEGDIR = fs::path(vm["segdir"].as<string>());
  fs::path SEGLIST = fs::path(vm["seglist"].as<string>());
  string POOLTYPE = vm["pool"].as<string>();
  bool NORMALIZE = vm["normalize"].as<bool>();
  bool FOREGROUND = vm["foreground"].as<bool>();
  cout << "FOREGROUND: " << FOREGROUND << endl;
  cout << "NORMALIZE: " << NORMALIZE << endl;
  long long START_IMGID = vm["startimgid"].as<long long>();
  int OUTTYPE = -1;
  if (vm["output-type"].as<string>().compare("text") == 0) {
    OUTTYPE = OUTTYPE_TEXT;
  } else if (vm["output-type"].as<string>().compare("lmdb") == 0) {
    OUTTYPE = OUTTYPE_LMDB;
  } else if (vm["output-type"].as<string>().compare("hdf5") == 0) {
    OUTTYPE = OUTTYPE_HDF5;
  } else {
    LOG(FATAL) << "Unknown output-type " << vm["output-type"].as<string>();
  }

  if (POOLTYPE.length() > 0) {
    LOG(INFO) << "Will be pooling with " << POOLTYPE;
  }

  Net<float> caffe_test_net(NETWORK_PATH.string(), caffe::TEST);
  caffe_test_net.CopyTrainedLayersFrom(MODEL_PATH.string());
  int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();

  // Get list of images in directory
  vector<fs::path> imgs;
  readList_withSpaces<fs::path>(IMGSLST, imgs);

  cout << "imgs  " << imgs.size() << endl;

  vector<fs::path> segpaths;
  if (SEGDIR.string().length() > 0 && (fs::exists(SEGDIR) || 
        SEGDIR.string().compare("service") == 0)) {
    LOG(INFO) << "Will be pruning the bounding boxes using "
              << "segmentation information";
    if (SEGLIST.string().length() > 0) {
      readList<fs::path>(SEGLIST, segpaths);
    } else {
      segpaths = imgs;
    }
  } else {
    SEGDIR = fs::path(""); // so that I don't need to check existance again
  }

  cout << "segpaths  " << segpaths.size() << endl;

  // Get list of image ids (1 indexed) to compute for 
  vector<long long> ids2compute4;
  if (vm["ids2compute4"].as<string>().length() > 0) {
    readList(vm["ids2compute4"].as<string>(), ids2compute4);
  } else if (vm["uniquelist"].as<string>().length() > 0) {
    getUniqueIds(fs::path(vm["uniquelist"].as<string>()), ids2compute4, START_IMGID);
  } else {
    for (long long imgid = START_IMGID; imgid <= imgs.size(); imgid++) {
      ids2compute4.push_back(imgid);
    }
  }

  std::shared_ptr<DiskVectorLMDB<vector<float>>> dv;
  if (OUTTYPE == OUTTYPE_LMDB) {
    dv = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
        new DiskVectorLMDB<vector<float>>(OUTDIR));
    // If it gets stuck here, it might be because the lock file is not
    // letting it update an existing store (maybe a unnatural exit last time).
    // Just delete the lock file (of course, if no other program is using it)
  }
  high_resolution_clock::time_point begin = high_resolution_clock::now();
  for (long long meta_i = 0; meta_i < ids2compute4.size(); meta_i++) {
    long long imgid = ids2compute4[meta_i];
    high_resolution_clock::time_point start = high_resolution_clock::now();
    fs::path imgpath = imgs[imgid - 1];
    cout << meta_i << " " << imgpath << endl;
    //cout << endl;
    //cout << "DEBUG: check100 " << meta_i << " " << ids2compute4.size() << endl;
    //cout << endl;
    if (meta_i % PRINT_INTERVAL == 0) {
      cout << "Doing for " << imgpath << " (" << meta_i << "/"
           << ids2compute4.size() << ")...";
    }

    vector<Mat> Is;
    Mat I;
    if (IMGSDIR.compare("service") == 0) {
      // read the image from the hbase based image read service
      readImageUsingService(imgpath.string(), I);
    } else {
      I = imread((IMGSDIR / imgpath).string());
    }
    Mat S; // get the segmentation image as well, used in debugging
    if (!I.data) {
      LOG(ERROR) << "Unable to read " << imgpath;
      continue;
    }
    vector<Rect> bboxes;
    bboxes.clear();
    if (WINDIR.string().size() > 0) {
      readBBoxesSelSearch<float>((WINDIR / (to_string((long long)imgid) + ".txt")).string(), bboxes);
    } else if (vm["sliding"].as<bool>()) {
      //  cout << "Debug: if 2 " << bboxes.size() << " " << vm["sliding"].as<bool>() << endl;
      genSlidingWindows(I.size(), bboxes);
    } else {
    //  cout << "Debug: if 1 " << bboxes.size() << endl;
      bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
    }
   // cout <<"DEBUG: bbox " << bboxes.size() << endl;
   // cout << endl;
    //cout << "DEBUG: check200 " << meta_i << endl;
    //cout << endl;
    // check if segdir defined. If so, then prune the list of bboxes
    if (SEGDIR.string().length() > 0) {
      if (SEGDIR.string().compare("service") == 0) {
        Mat S;
        readSegUsingService(I, S);
        cout << "DEBUG: check210 " << meta_i << endl;
        pruneBboxesWithSeg(I.size(), S, bboxes, FOREGROUND);
      } else {
        //cout << imgid << endl;
        //cout << segpaths.size() << endl;
        fs::path segpath = SEGDIR / segpaths[imgid - 1];
        if (!fs::exists(segpath)) {
          LOG(ERROR) << "Segmentation information not found for " << segpath;
        } else {
          //cout << "DEBUG: check250 " << meta_i << endl;
          pruneBboxesWithSeg(I.size(), segpath, bboxes, S, FOREGROUND);
        }
      }
    }
   // cout << endl;
   // cout << "DEBUG: check300 " << meta_i << endl;
   // cout << endl;
    LOG(INFO) << "Computing over " << bboxes.size() << " subwindows";
    // push in all subwindows
    for (int i = 0; i < bboxes.size(); i++) {
      Mat Itemp  = I(bboxes[i]);
      resize(Itemp, Itemp, Size(256, 256)); // rest of the transformation will run
                                            // like mean subtraction etc
      Is.push_back(Itemp);
    }
    // Uncomment to see the windows selected
    // DEBUG_storeWindows(Is, fs::path("temp/") / imgpath, I, S);

    // [layer[image[feature]]]
    vector<vector<vector<float>>> output;
    /**
     * Separately computing features for either case of text/hdf5 and lmdb because 
     * can using locking (and run parallel) for text output
     */
    if (OUTTYPE == OUTTYPE_TEXT || OUTTYPE == OUTTYPE_HDF5) {
      string fext = ".txt";
      if (OUTTYPE == OUTTYPE_HDF5) {
        fext = ".h5";
      }

      fs::path outFile = fs::change_extension(OUTDIR / imgpath, fext);
      if (output.size() > 1) {
        outFile = fs::change_extension(OUTDIR / fs::path(layers[0]) / imgpath, fext);
      }
      if (!lock(outFile)) {
        continue;
      }
      computeFeaturesPipeline(caffe_test_net, Is, layers, 
          BATCH_SIZE, output, false, POOLTYPE, NORMALIZE);
      for (int l = 0; l < output.size(); l++) {
        fs::path thisoutFile = fs::change_extension(OUTDIR / imgpath, fext);
        if (output.size() > 1) {
          thisoutFile = fs::change_extension(OUTDIR / fs::path(layers[l]) / imgpath, fext);
        }
        fs::create_directories(thisoutFile.parent_path());
        if (OUTTYPE == OUTTYPE_TEXT) {
          dumpFeatures_txt(thisoutFile, output[l]);
        } else {
          dumpFeatures_hdf5(thisoutFile, output[l]);
        }
      }
      unlock(outFile);
    } else if (OUTTYPE == OUTTYPE_LMDB) {
      if (layers.size() > 1) {
        LOG(FATAL) << "Multiple layer output is not suppported with lmdb output.";
      }
      computeFeaturesPipeline(caffe_test_net, Is, layers, 
          BATCH_SIZE, output, false, POOLTYPE, NORMALIZE);
      // output into a DiskVector
      for (int i = 0; i < output[0].size(); i++) {
        dv->Put(hashCompleteName(imgid, i), output[0][i]);
      }
    }
    if (meta_i % PRINT_INTERVAL == 0) {
      high_resolution_clock::time_point end = high_resolution_clock::now();
      cout << "Done in " << duration_cast<milliseconds>(end - start).count()
           << "ms" << endl
           << "Average taking " 
           << duration_cast<milliseconds>(end - begin).count() * 1.0f / 
              (meta_i + 1) << "ms" << endl;
    }
  }

  return 0;
}

inline void dumpFeatures_txt(const fs::path& fpath, const vector<vector<float>>& feats) {
  FILE* fout = fopen(fpath.string().c_str(), "w");
  for (int fi = 0; fi < feats.size(); fi++) {
    for (int i = 0; i < feats[fi].size(); i++) {
      if (feats[fi][i] == 0) {
        fprintf(fout, "0 ");
      } else {
        fprintf(fout, "%f ", feats[fi][i]);
      }
    }
    fprintf(fout, "\n");
  }
  fclose(fout);
}

inline void dumpFeatures_hdf5(const fs::path& fpath, vector<vector<float>>& feats) {
  if (feats.size() == 0) {
    cerr << "No items in the feature. Not writing the file." << endl;
    return;
  }
  int DIM1 = feats.size();
  int DIM2 = feats[0].size();
  float* feats_raw = new float[DIM1 * DIM2];
  for (int i = 0; i < feats.size(); i++) {
    for (int j = 0; j < feats[i].size(); j++) {
      feats_raw[i * DIM2 + j] = feats[i][j];
    }
  }
  hsize_t dimsf[2];
  dimsf[0] = DIM1; dimsf[1] = DIM2;
  hsize_t cdims[2];
  cdims[0] = 1; cdims[1] = DIM2;
  hid_t file = H5Fcreate(fpath.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
  hid_t dataspace = H5Screate_simple(2, dimsf, NULL);
  hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
  H5Pset_deflate(dcpl, 6);
  H5Pset_chunk(dcpl, 2, cdims);
  hid_t dataset = H5Dcreate(file, "feats", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, dcpl,
      H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, feats_raw);

  delete[] feats_raw;
  H5Sclose(dataspace);
  H5Dclose(dataset);
  H5Fclose(file);
}

inline long long hashCompleteName(long long imgid, int id) { // imgid is 1 indexed, id is 0 indexed
  // both will be 1 indexed <convention set Saturday 11 April 2015 01:57:20 AM GMT>
  return imgid * MAXFEATPERIMG + id + 1;
}

void readImageUsingService(const string& imid, Mat& I) {
  const int port_num = 5554;
  static bool initialized = false;
  static zmq::context_t *context = NULL;
  static zmq::socket_t *socket = NULL;
  if (! initialized) {
    context = new zmq::context_t(1);
    socket = new zmq::socket_t(*context, ZMQ_REQ);
    socket->connect(("tcp://localhost:" + to_string((long long)port_num)).c_str());
    initialized = true;
  }
  zmq::message_t request(imid.length());
  memcpy((void *) request.data(), imid.c_str(), imid.length());
  socket->send(request);

  zmq::message_t reply;
  socket->recv(&reply);
  char *reply_data = static_cast<char*>(reply.data());
  vector<char> reply_vector_str(reply_data, reply_data + reply.size());
  try {
    I = imdecode(reply_vector_str, CV_LOAD_IMAGE_COLOR);
  } catch (cv::Exception& e) {
    LOG(ERROR) << "Unable to read image at " << imid << ". Returning black image... ";
    I = Mat(100, 100, CV_8UC3, Scalar(0,0,0));
  }
}

void readSegUsingService(const Mat& I, Mat& S) {
  const int port_num = 5556;
  const string TEMP_SEG_DIR = "/tmp/segtemp/computeFeatures/";
  const string qimg_path = TEMP_SEG_DIR + "img.jpg";
  const string res_path = "/srv2/rgirdhar/Work/Code/0005_ObjSegment/scripts/service_scripts/temp-dir/result.jpg"; 

  static bool initialized = false;
  static zmq::context_t *context = NULL;
  static zmq::socket_t *socket = NULL;
  if (! initialized) {
    context = new zmq::context_t(1);
    socket = new zmq::socket_t(*context, ZMQ_REQ);
    socket->connect(("tcp://localhost:" + to_string((long long)port_num)).c_str());
    initialized = true;
  }
  imwrite(qimg_path.c_str(), I);
  zmq::message_t request(qimg_path.length());
  memcpy((void *) request.data(), qimg_path.c_str(), qimg_path.length());
  socket->send(request);

  zmq::message_t reply;
  socket->recv(&reply);
  S = imread(res_path, CV_LOAD_IMAGE_GRAYSCALE);
}

