#include <iostream>
#include <chrono>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <glog/logging.h>
#include "storage/DiskVectorLMDB.hpp"
#include "LSH.hpp"
#include "Resorter.hpp"
#include "utils.hpp"
#include "lock.hpp"
#include "config.hpp"

using namespace std;
using namespace std::chrono;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

long long getIndex(long long, int); // both must be 1 indexed
long long getIndex_DEPRECATED(long long, int); // both must be 1 indexed
void readUniqueList(const fs::path& fpath, vector<int>& imgIds);
void generateTrainData(const vector<int>&, const vector<int>&, 
    const DiskVectorLMDB<vector<float>>&, vector<vector<float>>&, 
    bool deprecated_stor, int nTrain);

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("datapath,d", po::value<string>()->required(),
     "Path to LMDB where the data is stored")
    ("imgslist,n", po::value<string>()->required(),
     "Filenames of all images in the corpus")
    ("ids2compute4", po::value<string>()->default_value(""),
     "File with indexes (1-indexed) of all images to be added to table")
    ("featcount,c", po::value<string>()->default_value(""),
     "File with list of number of features in each image."
     "NOT correspoding to above list, but for all images in global imgslist")
    ("load,l", po::value<string>(),
     "Path to load the initial hash table")
    ("save,s", po::value<string>(),
     "Path to save the hash table")
    ("nbits,b", po::value<int>()->default_value(250),
     "Number of bits in the representation")
    ("ntables,t", po::value<int>()->default_value(15),
     "Number of random proj tables in the representation")
    ("saveafter,a", po::value<int>()->default_value(1800), // every 1/2 hour
     "Time after which to snapshot the model (seconds)")
    ("printafter", po::value<int>()->default_value(5), // every 5 seconds
     "Time after which to print output (seconds)")
    ("nTrain", po::value<int>()->default_value(100000), // 100K (8GB pool5) by default
     "Number of random elements from imgComputeIDs to be used to train ITQ. "
     "Note that this mainly depends on the memory available.")
    ("deprecated-stor", po::bool_switch()->default_value(false),
     "The data store being used is using the deprecated naming, "
     "i.e. 0 indexed imids and featids. Note that the model generated "
     "will still use 1 indexed")
    ("duplist", po::value<fs::path>()->default_value(""),
     "Path to list with unique/duplicate entries. Will build index "
     "only for entries which start with 'U'")
    ("compressedFeatStor", po::bool_switch()->default_value(false),
     "Set this flag if the data store contains compressed features")
    ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    cerr << desc << endl;
    return -1;
  }
  try {
    po::notify(vm);
  } catch(po::error& e) {
    cerr << e.what() << endl;
    return -1;
  }
  
  // read the list of images to hash
  int saveafter = vm["saveafter"].as<int>();
  int printafter = vm["printafter"].as<int>();
  int nTrain = vm["nTrain"].as<int>();
  //nTrain = 100;
  cout << "nTrain: " << nTrain << endl;
  bool deprecated_stor = vm["deprecated-stor"].as<bool>();
  int imgslst_size = countNewlines(vm["imgslist"].as<string>());
  vector<int> featcounts(imgslst_size, 1); // default: 1 feat/image
  if (vm["featcount"].as<string>().length() > 0) {
    featcounts.clear();
    readList(vm["featcount"].as<string>(), featcounts);
  }
  vector<int> imgComputeIds;
  if (vm["ids2compute4"].as<string>().length() > 0) {
    readList(vm["ids2compute4"].as<string>(), imgComputeIds);
  } else if (vm["duplist"].as<fs::path>().string().length() > 0) {
    readUniqueList(vm["duplist"].as<fs::path>(), imgComputeIds);
  } else {
    // all images
    for (int i = 1; i <= imgslst_size; i++) {
      imgComputeIds.push_back(i);
    }
  }
  
  DiskVectorLMDB<vector<float>> tree(vm["datapath"].as<string>(), 1, vm["compressedFeatStor"].as<bool>());

  std::shared_ptr<LSH> l(new LSH(vm["nbits"].as<int>(), vm["ntables"].as<int>()));
  if (vm.count("load")) {
    ifstream ifs(vm["load"].as<string>(), ios::binary);
    boost::archive::binary_iarchive ia(ifs);
    ia >> *l;
    cout << "Loaded the search model for update" << endl;
  } else {
    vector<vector<float>> trainData;
    generateTrainData(imgComputeIds, featcounts, tree, trainData, deprecated_stor, nTrain);
    cout << "Generated " << trainData.size() 
         << " training data. Starting to train..." << endl;
    cout << "nTrain: " << nTrain << endl;
    l->train(trainData);
  }
  vector<float> feat;
  
  high_resolution_clock::time_point last_print, last_save;
  last_print = last_save = high_resolution_clock::now();

  if (l->lastLabelInserted >= 0) {
    cout << "Ignoring uptil (and including) " << l->lastLabelInserted 
         << ". Already exists in the index" << endl;
  }
  for (int meta_i = 0; meta_i < imgComputeIds.size(); meta_i++) {
    int i = imgComputeIds[meta_i] - 1; // hash this image
    for (int j = 0; j < featcounts[i]; j++) {
      long long idx;
      if (deprecated_stor) {
        idx = getIndex_DEPRECATED(i+1, j+1);
      } else {
        idx = getIndex(i+1, j+1);
      }
      if (l->lastLabelInserted >= idx) {
        continue;
      }
      if (!tree.Get(idx, feat)) break;
      l->insert(feat, idx);
    }
    high_resolution_clock::time_point now = high_resolution_clock::now();
    if (duration_cast<seconds>(now - last_print).count() >= printafter) {
      cout << "Done for " << meta_i + 1  << "/" << imgComputeIds.size()
           << " in " 
           << duration_cast<milliseconds>(now - last_print).count()
           << "ms" <<endl;
      last_print = now;
    }
    if (duration_cast<seconds>(now - last_save).count() >= saveafter) {
      if (vm.count("save")) {
        cout << "Saving model to " << vm["save"].as<string>() << "...";
        cout.flush();
        ofstream ofs(vm["save"].as<string>(), ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << *l;
        cout << "done." << endl;
      }
      last_save = now;
    }
  }

  if (vm.count("save")) {
    cout << "Saving model to " << vm["save"].as<string>() << "...";
    cout.flush();
    ofstream ofs(vm["save"].as<string>(), ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << *l;
    cout << "done." << endl;
  }

  return 0;
}

long long getIndex(long long imid, int pos) { // imid and pos must be 1 indexed
  return imid * MAXFEATPERIMG + pos;
}

long long getIndex_DEPRECATED(long long imid, int pos) { // imid and pos must be 1 indexed
  return (imid - 1) * MAXFEATPERIMG + (pos - 1);
}

void readUniqueList(const fs::path& fpath, vector<int>& imgIds) {
  imgIds.clear();
  ifstream fin(fpath.string());
  string line;
  int lno = 1;
  while (getline(fin, line)) {
    if (line[0] == 'U') {
      imgIds.push_back(lno);
    }
    lno++;
  }
  fin.close();
}

void generateTrainData(const vector<int>& imgComputeIDs, const vector<int>& featcounts,
    const DiskVectorLMDB<vector<float>>& tree, vector<vector<float>>& outputTrainData,
    bool deprecated_stor, int nTrain) {
  // select randomly nTrain elements from all patches and push into outputTrainData
  vector<long long> allfeatids;
  for (int i = 0; i < imgComputeIDs.size(); i++) {
    for (int j = 1; j <= featcounts[imgComputeIDs[i] - 1]; j++) {
      long long imid;
      if (deprecated_stor) {
        imid = getIndex_DEPRECATED(imgComputeIDs[i], j);
      } else {
        imid = getIndex(imgComputeIDs[i], j);
      }
      allfeatids.push_back(imid);
    }
  }
  random_shuffle(allfeatids.begin(), allfeatids.end());
  nTrain = min(nTrain, (int) allfeatids.size());
  for (int i = 0; i < nTrain; i++) {
    vector<float> feat;
    if (tree.Get(allfeatids[i], feat)) {
      outputTrainData.push_back(feat);
    } // else it was not able to retrieve, so forget about it 
  }
}
