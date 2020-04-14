/**
 * This is now deprecated for building index. 
 * Use buildIndex.cpp now
 */

#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
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
  google::InitGoogleLogging(argv[0]);
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("datapath,d", po::value<string>()->required(),
     "Path to leveldb where the data is stored")
    ("dimgslist,n", po::value<string>()->required(),
     "File with list of all images")
    ("featcount,c", po::value<string>()->default_value(""),
     "File with list of number of features in each image")
    ("outdir,o", po::value<string>(),
     "Output directory to store output matches")
    ("qimgslist,m", po::value<string>(),
     "File with list of all query images")
    ("save,s", po::value<string>(),
     "Path to save the hash table")
    ("load,l", po::value<string>(),
     "Path to load the hash table from")
    ("topk,k", po::value<int>()->default_value(30),
     "Top-K elements to output after search")
    ("nbits,b", po::value<int>()->default_value(250),
     "Number of bits in the representation")
    ("ntables,t", po::value<int>()->default_value(15),
     "Number of random proj tables in the representation")
    ("bruteforce,z", po::bool_switch()->default_value(false),
     "Use brute force search over all the features. (Use with caution)")
    ("updateres,u", po::bool_switch()->default_value(false),
     "Update the results. This will read the existing results, and not modify"
     "it if it has been computed for any patch")
    ("boxes2run4", po::value<string>()->default_value(""),
     "Path to directory with files with list of selsearch ids to run for. "
     "Each list must be 1-indexed.")
    ("reranklimit", po::value<int>()->default_value(-1),
     "Limit the number of matching feature that will be reranked."
     "If there are more than this number, that bounding box will be ignored"
     "(no results outputted). Default = -1 => no limits")
    ("nRerank", po::value<int>()->default_value(5000),
     "Number of images from the pre-ranking to re-sort")
    ("simMetric", po::value<string>()->default_value("cosine"),
     "Pick similarity metric to be used for resorting [cosine/euclidean]")
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
 
  int nRerank = vm["nRerank"].as<int>();
  // read the list of images to hash
  vector<fs::path> imgslst;
  readList(vm["dimgslist"].as<string>(), imgslst);
  vector<unsigned> featcounts(imgslst.size(), 1); // default: 1 feat/image
  if (vm["featcount"].as<string>().length() > 0) {
    featcounts.clear();
    readList(vm["featcount"].as<string>(), featcounts);
  }
  
  bool BFORCE = vm["bruteforce"].as<bool>();
  int simMetric;
  if (vm["simMetric"].as<string>().compare("cosine") == 0) {
    simMetric = SIM_METRIC_COSINE;
  } else if (vm["simMetric"].as<string>().compare("euclidean") == 0) {
    simMetric = SIM_METRIC_EUCLIDEAN;
  } else {
    cerr << "Distance metric not implemented!!!" << endl;
    return -1;
  }
  unordered_set<long long int> searchspace;
  LSH *l = NULL;
  if (BFORCE) {
    cerr << "Caution: Running brute force search...";
    if (featcounts.size() == 0) {
      cerr << "Brute force requires feature counts. Exitting..." << endl;
      return -1;
    }
    getAllSearchspace(featcounts, searchspace);
  } else if (vm.count("load")) {
    cout << "Loading model from " << vm["load"].as<string>() << "...";
    cout.flush();
    ifstream ifs(vm["load"].as<string>(), ios::binary);
    boost::archive::binary_iarchive ia(ifs);
    l = new LSH(0,0); // need to create this dummy obj, don't know how else...
    ia >> *l;
    cout << "done." << endl;
  } else if (vm.count("datapath")) {
    cerr << "Use buildIndex to create the index." << endl;
    return -1;
  }

  if (vm.count("save")) {
    cout << "Saving model to " << vm["save"].as<string>() << "...";
    cout.flush();
    ofstream ofs(vm["save"].as<string>(), ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << *l;
    cout << "done." << endl;
  }

  if (vm.count("qimgslist") && vm.count("outdir")) {
    vector<int> qlist;
    readList(vm["qimgslist"].as<string>(), qlist);
    auto featstor = shared_ptr<DiskVectorLMDB<vector<float>>>(
        new DiskVectorLMDB<vector<float>>(vm["datapath"].as<string>(), 1));

    for (int i = 0; i < qlist.size(); i++) {
      // by default, search on all features in the image.
      fs::path fpath = fs::path(vm["outdir"].as<string>()) /
          fs::path(to_string(static_cast<long long>(qlist[i])) + ".txt");
      if (!lock(fpath)) {
        if (vm["updateres"].as<bool>()) {
          cerr << "Update results specified. Trying to get update lock...";
          if (lock(fpath, true)) {
            cerr << "UpdateLock acquired" << endl;
          } else {
            cerr << "Skipping " << fpath << "..." << endl;
            continue;
          }
        } else {
          cerr << "Skipping " << fpath << "..." << endl;
          continue;
        }
      }

      vector<vector<pair<float, long long int>>> allres{featcounts[qlist[i] - 1]};
      if (vm["updateres"].as<bool>()) {
        // read in the current status of results
        readResults(fpath, allres);
      }

      // read list of boxes
      vector<int> boxes2run4; // 1 indexed
      if (vm["boxes2run4"].as<string>().length() > 0) {
        readList(vm["boxes2run4"].as<string>() + "/" + to_string((long long) qlist[i]) + ".txt", boxes2run4);
      } else {
        for (int j = 0; j < featcounts[qlist[i] - 1]; j++) {
          boxes2run4.push_back(j + 1);
        }
      }

      high_resolution_clock::time_point last_print = high_resolution_clock::now();
      for (int meta_j = 0; meta_j < boxes2run4.size(); meta_j++) {
        int j = boxes2run4[meta_j] - 1; // 0 indexed j
        if (allres[j].size() > 0) {
          // means already solved (maybe read from output file in update step)
          continue;
        }
        vector<pair<float,long long int>> res;
        #if defined(RAND_SAMPLE)
          // randomly keep only 1000 of the windows (since can't do for all of them!)
          float perc = (RAND_SAMPLE * 1.0f) / featcounts[i];
          if ((double) rand() / RAND_MAX > perc) {
            allres[j] = res;
            continue;
          }
        #endif

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        int idx = (qlist[i]) * MAXFEATPERIMG + j + 1;
        vector<float> feat;
        if (!featstor->Get(idx, feat)) {
          cerr << "Ignoring..." << endl;
          allres[j] = res;
          continue;
        }
        
        unordered_set<long long int> temp;
        if (BFORCE) {
          temp = searchspace;
        } else {
          l->search(feat, temp, nRerank);
        }
        
        // enforce limit on number of features that get re-ranked
        int rerank_limit = vm["reranklimit"].as<int>();
        if (rerank_limit >= 0 && temp.size() > rerank_limit) {
          allres[j] = res;
          continue;
        }
        
        Resorter::resort_multicore(temp, featstor, feat, res, simMetric);
        allres[j] = vector<pair<float, long long int>>(res.begin(), 
            min(res.begin() + vm["topk"].as<int>(), res.end()));
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        if (duration_cast<seconds>(t2 - last_print).count() >= 2) {
          auto duration = duration_cast<milliseconds>(t2 - t1).count();
          cout << "Search done for " << qlist[i] << ":" << j << " in " << duration 
               << " ms (re-ranked: " << temp.size() << ")" << endl;
          cout.flush();
          last_print = t2;
        }
      }
      ofstream fout(fpath.string());
      for (auto res = allres.begin(); res != allres.end(); res++) {
        int pos = 0;
        for (auto it = res->begin(); it != res->end(); it++) {
          fout << it->second << ":" << it->first << " ";
        }
        fout << endl;
      }
      fout.close();
      unlock(fpath);
    }
  }

  return 0;
}

