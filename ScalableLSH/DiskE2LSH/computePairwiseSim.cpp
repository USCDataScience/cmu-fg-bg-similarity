#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <Eigen/Dense>
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

int main(int argc, char* argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("datapath,d", po::value<string>()->required(),
     "Path to LMDB where the data is stored")
    ("imgidslist,l", po::value<string>()->required(),
     "File with list of image ids to run this for (this lst shd be 1 indexed)")
    ("featcount,c", po::value<string>()->required(),
     "File with list of number of features in each image")
    ("outdir,o", po::value<string>(),
     "Output directory to store output nxn matches")
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
  vector<int> imgidslist;
  readList(vm["imgidslist"].as<string>(), imgidslist);
  vector<int> featcounts; // default: 1 feat/image
  readList(vm["featcount"].as<string>(), featcounts);

  auto featstor = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
      new DiskVectorLMDB<vector<float>>(vm["datapath"].as<string>(), 1));

  for (int i = 0; i < imgidslist.size(); i++) {
    int imgid = imgidslist[i];
    fs::path outfpath = fs::path(vm["outdir"].as<string>()) / 
      fs::path(to_string((long long)imgid) + ".txt");
    
    if (!lock(outfpath)) {
      continue;
    }

    Eigen::MatrixXf sims;
    Resorter::computePairwiseSim(featstor, imgid, featcounts[imgid - 1], sims);
    ofstream fout(outfpath.string());
    fout << sims;
    fout.close();

    unlock(outfpath);
  }

  return 0;
}

