#include "DiskVector.hpp"
#include "DiskVectorLMDB.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

//#define FPATH "/IUS/vmr105/rohytg/data/selsearch_feats_all_normalized.txt"
#define FPATH "dummy.txt"
//#define FPATH "marked_feats_all.txt"
//#define FPATH "/home/rgirdhar/Work/Projects/001_DetectionRetrieval/BgMatchesObjDet/tempdata/marked_feats_all.txt"

void normalize(vector<float>& feat) {
  float norm = 0;
  for (auto it = feat.begin(); it != feat.end(); it++) {
    norm += *it * (*it);
  }
  norm = sqrt(norm);
  for (auto it = feat.begin(); it != feat.end(); it++) {
    *it = *it / norm; 
  }

}

void readAndIndex(fs::path fpath) {
  DiskVectorLMDB<vector<float>> d("selsearch_feats");
  ifstream ifs(fpath.string().c_str(), ios::in);
  string line;
  float el;
  int i = 0;
  while (getline(ifs, line)) {
    vector<float> feat;
    istringstream iss(line);
    while (iss >> el) {
      feat.push_back(el);
    }
    normalize(feat);
    d.Put(i, feat);
    i++;
    cout << "done for " << i << endl;
  }
}

int main() {
  readAndIndex(FPATH);
  return 0;
}

