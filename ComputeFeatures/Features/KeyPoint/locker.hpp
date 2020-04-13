#include <sys/stat.h>
#include <sys/types.h>
#include <string>
using namespace std;

namespace Locker {
  bool lock(const string &fpath) {
    struct stat info;
    string lock_fpath = fpath + ".lock";
    if (stat(fpath.c_str(), &info) != 0 && 
        stat(lock_fpath.c_str(), &info) != 0) {
      mkdir(lock_fpath.c_str(),
          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      return true;
    } else {
      return false;
    }
  }

  void unlock(const string &fpath) {
    string lock_fpath = fpath + ".lock";
    rmdir(lock_fpath.c_str());
  }
}
