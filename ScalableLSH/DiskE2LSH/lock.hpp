#ifndef LOCK_HPP
#define LOCK_HPP

#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

/**
 * Function to lock the access to file. Return true if previously unlocked and now able to
 * lock
 */
// set updateLock = true if you want a lock even if the output file exists
// This is useful when you want to update files
bool lock(fs::path fpath, bool updateLock = false) {
    fs::path lock_fpath = fs::path(fpath.string() + ".lock");
    // if updateLock specified, make a lock even if the file exists
    if (fs::exists(lock_fpath) || (!updateLock && fs::exists(fpath))) {
      return false;
    }
    fs::create_directories(lock_fpath);
    return true;
}

bool unlock(fs::path fpath) {
    fs::path lock_fpath = fs::path(fpath.string() + ".lock");
    if (!fs::exists(lock_fpath) || !fs::exists(fpath)) return false;
    return fs::remove(lock_fpath);
}

#endif

