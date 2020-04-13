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

using namespace std;
namespace fs = boost::filesystem;

/**
 * This class stores a vector<T> at every position (hence is actually a 2D vector).
 * Disk based storage powered by OpenLDAP's MDB
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

  public:
  DiskVectorLMDB(fs::path _fpath, bool _rdonly = false) : 
    fpath(_fpath), putcount(0), rdonly(_rdonly) {
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
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

  bool Get(long long pos, T& output) const {
    output.clear();
    MDB_val key, data;
    string pos_s = to_string(pos);
    key.mv_size = pos_s.size();
    key.mv_data = reinterpret_cast<void*>(&pos_s[0]);
    int rc = mdb_get(mdb_txn, mdb_dbi, &key, &data);
    if (rc == MDB_NOTFOUND) {
      cerr << "Unable to read element at " << pos << endl;
      return false;
    }
    char *cstr = new char[data.mv_size]; // TODO: avoid re-alloc everytime
    memcpy(cstr, data.mv_data, data.mv_size);
    string str(cstr, data.mv_size);
    istringstream iss(str);
    boost::archive::binary_iarchive ia(iss);
    ia >> output;
    delete[] cstr;
    return true;
  }

  bool Put(long long pos, const T& input) {
    ostringstream oss;
    boost::archive::binary_oarchive oa(oss);
    oa << input;
    string hash = oss.str();
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
};

#endif

