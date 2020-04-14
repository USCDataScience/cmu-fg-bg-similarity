/**
 * Code to compress an existing DiskVectorLMDB store
 * (c) Rohit Girdhar
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "../DiskVectorLMDB.hpp"

using namespace std;
using namespace std::chrono;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int
main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("infeatstor,i", po::value<fs::path>()->required(),
     "Path to feature store to compress")
    ("outfeatstor,o", po::value<fs::path>()->required(),
     "Path to output feature store")
    ("keyslist,k", po::value<fs::path>()->required(),
     "File with images list")
    ("startpos,s", po::value<unsigned long long>()->default_value(1),
     "Position to start reading the file [1 indexed]")
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
  
  unsigned long long start_pos = vm["startpos"].as<unsigned long long>();
  fs::path infeatstor_fpath = vm["infeatstor"].as<fs::path>();
  fs::path outfeatstor_fpath = vm["outfeatstor"].as<fs::path>();
  auto infeatstor = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
      new DiskVectorLMDB<vector<float>>(infeatstor_fpath.string(), 
        /* readonly= */ 1,
        /* compress= */ 0));
  auto outfeatstor = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
      new DiskVectorLMDB<vector<float>>(outfeatstor_fpath.string(),
        /* readonly= */ 0,
        /* compress= */ 1));

  ifstream fin(vm["keyslist"].as<fs::path>().string());
  if (! fin.is_open()) {
    cerr << "Unable to open the file " << vm["keyslist"].as<fs::path>() << endl;
  }
  long long key;
  high_resolution_clock::time_point start_time =
    high_resolution_clock::now();
  unsigned long long lno = 0;
  while (fin >> key) {
    lno++;
    if (lno < start_pos) {
      continue;
    }
    string feat = infeatstor->directGet(key);
    if (feat.length() < 0 || ! outfeatstor->directPut(key, feat)) {
      cerr << "Couldn't read/write " << key << endl;
    }
    high_resolution_clock::time_point cur_time =
      high_resolution_clock::now();
    if (duration_cast<seconds>(cur_time - start_time).count() > 2) {
      cout << "Done for " << lno << endl;
      start_time = cur_time;
    }
  }
  fin.close();

  return 0;
}

