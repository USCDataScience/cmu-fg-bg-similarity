#include "DiskVectorLMDB.hpp"
#include "DiskVector.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <chrono>

using namespace std;
using namespace std::chrono;
namespace fs = boost::filesystem;

void readAndPrint() {
  DiskVectorLMDB<vector<float>> d("/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_pool5_uni_normed_LMDB", 1);
  DiskVector<vector<float>> d2("/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_pool5_unified");
  vector<float> temp, temp2;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = 1; i < 1000; i += 10) {
    d.Get(i, temp);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now(); 
  for (int i = 1; i < 1000; i += 10) {
    d2.Get(i, temp2);
  }
  high_resolution_clock::time_point t3 = high_resolution_clock::now();
  cout << "LMDB: " << duration_cast<microseconds>(t2 - t1).count() << "us. Leveldb: " << duration_cast<microseconds>(t3 - t2).count() << "us" << endl;
}

int main() {
  readAndPrint();
  return 0;
}

