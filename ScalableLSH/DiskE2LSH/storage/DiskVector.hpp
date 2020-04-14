#ifndef DISKVECTOR_HPP
#define DISKVECTOR_HPP

#include <leveldb/db.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

/**
 * This class stores a vector<T> at every position (hence is actually a 2D vector).
 * Disk based storage powered by Google's leveldb
 */
template<typename T>
class DiskVector {
  fs::path fpath; // path to the disk storage
  leveldb::DB *db;
    
public:
  DiskVector(fs::path _fpath) {
    fpath = _fpath;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::Status s = leveldb::DB::Open(options, fpath.string(), &db);
    if (!s.ok()) {
      cerr << s.ToString() << endl;
    }
  }

  ~DiskVector() {
    delete db;
  }

  bool Get(long long pos, T& output) const {
    output.clear();
    // read from the leveldb
    string value;
    leveldb::Slice key((char*)&pos, sizeof(long long));
    leveldb::Status st = db->Get(leveldb::ReadOptions(), key, &value);
    if (!st.ok()) {
      cerr << "Unable to read elt at " << pos << " due to " << st.ToString();
      return false;
    }
    istringstream iss(value);
    boost::archive::binary_iarchive ia(iss);
    ia >> output;
    return true;
  }

  bool Put(long long pos, const T& input) {
    ostringstream oss;
    boost::archive::binary_oarchive oa(oss);
    oa << input;
    leveldb::Slice key((char*)&pos, sizeof(long long));
    leveldb::Status st = db->Put(leveldb::WriteOptions(), key, oss.str());
    if (!st.ok()) {
      cerr << "Unable to write at pos " << pos << " due to " << st.ToString();
      return false;
    }
    return true;
  }
};

#endif

