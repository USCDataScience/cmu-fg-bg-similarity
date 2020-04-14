#ifndef DISKVECTOR_UTILS_HPP
#define DISKVECTOR_UTILS_HPP

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <glog/logging.h>

using namespace std;

template<typename Dtype>
void unhashObj(Dtype& obj, const string& hash) {
  istringstream iss(hash);
  boost::archive::binary_iarchive ia(iss);
  ia >> obj;
}

template<typename Dtype>
void hashObj(const Dtype& obj, string& hash) {
  ostringstream oss;
  boost::archive::binary_oarchive oa(oss);
  oa << obj;
  hash = oss.str();
}

#endif

