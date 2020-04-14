#ifndef DISKVECTORLMDB_HPP
#define DISKVECTORLMDB_HPP

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <lmdb.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <chrono>
#include "zlib_utils.hpp"

using namespace std;
using namespace std::chrono;
namespace fs = boost::filesystem;

/**
 * This class stores a vector<T> at every position
 * (hence is actually a 2D vector).
 * Disk based storage powered by OpenLDAP's MDB
 * NOTE: If this gets stuck at mdb_open command or so while updating,
 * it might be because of the lock file. Just delete it.
 * TODO: Upgrade to use nicer C++ interface to LMDB: https://github.com/bendiken/lmdbxx
 */
template<typename T>
class DiskVectorLMDB {
  MDB_env* mdb_env;
  MDB_dbi mdb_dbi;
  MDB_txn* mdb_txn;
  MDB_val mdb_key, mdb_data;
  fs::path fpath; // path to the disk storage
  int putcount;
  int rdonly;
  bool compress;

  public:
  DiskVectorLMDB(fs::path _fpath, bool _rdonly = false, bool _compress = false) : 
    fpath(_fpath), putcount(0), rdonly(_rdonly), compress(_compress) {
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 2048000000000), MDB_SUCCESS);  // 2TB
    // mapsize defines the max size of database, so keep it large
    //CHECK_EQ(mdb_env_set_mapsize(mdb_env, 2000000000), MDB_SUCCESS);  // 2GB
    int READ_FLAG = rdonly ? MDB_RDONLY : 0;
    if (!rdonly) {
      if (!fs::is_directory(_fpath)) {
        CHECK_EQ(mkdir(fpath.string().c_str(), 0744), 0);
      } else {
        cerr << "Warning: A folder already exists at "
             << _fpath << ". Trying to update that...";
      }
    }
    CHECK_EQ(mdb_env_open(mdb_env, fpath.string().c_str(), READ_FLAG, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, READ_FLAG, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
      << "mdb_open failed";
  }

  ~DiskVectorLMDB() {
    if (!rdonly)
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(mdb_env, mdb_dbi);
    if (rdonly)
      mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);
  }

  string directGet(long long pos) const {
    MDB_val key, data;
    string pos_s = to_string(pos);
    key.mv_size = pos_s.size();
    key.mv_data = reinterpret_cast<void*>(&pos_s[0]);
    int rc = mdb_get(mdb_txn, mdb_dbi, &key, &data);
    if (rc == MDB_NOTFOUND) {
      cerr << "Unable to read element at " << pos << endl;
      return "";
    }
    char *cstr = new char[data.mv_size];
    memcpy(cstr, data.mv_data, data.mv_size);
    string str(cstr, data.mv_size);
    if (compress) {
      // only for debugging. It takes about 0.5ms per decompress
      // high_resolution_clock::time_point st = high_resolution_clock::now();
      str = zlib_decompress_string(str);
      // high_resolution_clock::time_point end = high_resolution_clock::now();
      // cout << "Time to decompress: " << duration_cast<nanoseconds>(end - st).count() << "ns" << endl;
    }
    delete[] cstr;
    return str;
  }

  bool Get(long long pos, T& output) const {
    output.clear();
    string str = directGet(pos);
    if (str.size() == 0) {
      return false;
    }
    istringstream iss(str);
    boost::archive::binary_iarchive ia(iss);
    ia >> output;
    return true;
  }

  bool directPut(long long pos, const string& input) {
    string hash = input;
    if (compress) {
      hash = zlib_compress_string(hash);
    }
    mdb_data.mv_size = hash.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&hash[0]);
    string pos_s = to_string(pos);
    mdb_key.mv_size = pos_s.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&pos_s[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
      << "mdb_put failed";
    if (++putcount % 1000 == 0) {
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
        << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    }
    return true;
  }

  bool Put(long long pos, const T& input) {
    ostringstream oss;
    boost::archive::binary_oarchive oa(oss);
    oa << input;
    string hash = oss.str();
    return directPut(pos, hash);
  }
};

#endif

