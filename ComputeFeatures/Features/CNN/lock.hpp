#ifndef LOCK_HPP
#define LOCK_HPP

#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

/**
 * Function to lock the access to file. Return true if previously unlocked and now able to
 * lock
 */
bool lock(fs::path fpath) {
    fs::path lock_fpath = fs::path(fpath.string() + ".lock");
    if (fs::exists(fpath) || fs::exists(lock_fpath)) return false;
    fs::create_directories(lock_fpath);
    return true;
}

bool unlock(fs::path fpath) {
    fs::path lock_fpath = fs::path(fpath.string() + ".lock");
    if (!fs::exists(lock_fpath)) return false;
    return fs::remove(lock_fpath);
}

#endif

