#ifndef LSHFUNC_HPP
#define LSHFUNC_HPP

#ifndef EIGEN_CONFIG_H_
#define EIGEN_CONFIG_H_

#include <boost/serialization/array.hpp>
// w.r.t Eigen_3.2.4/Eigen/Core
#define EIGEN_DENSEBASE_PLUGIN "../../../../EigenDenseBaseAddons.hpp"
#include <Eigen/Core>
#endif // EIGEN_CONFIG_H_


#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <cmath>
#include <functional>
#include "config.hpp"

using namespace std;
namespace fs = boost::filesystem;

class LSHFunc {
  float w;
  int k; // number of bits in a function (length of key)
  int dim; // dimension of features
  Eigen::MatrixXf A;
  Eigen::MatrixXf b;
  
public:
  LSHFunc(int _k, int _dim): k(_k), dim(_dim) {
    genLSHfunc();
  }
  LSHFunc() {} // used while serializing

  void genLSHfunc() {
    w = 24; // default value
    A = Eigen::MatrixXf::Random(dim, k); // TODO: Use normal distribution to sample (as in GS code)
    typedef boost::mt19937 RNGType;
    RNGType rng;
    boost::uniform_real<> generator(0, w);
    boost::variate_generator<RNGType, boost::uniform_real<>> dice(rng, generator);

    b = Eigen::MatrixXf::Random(1, k);
    for (int i = 0; i < k; i++) {
      b(0, i) = dice();
    }
  }

  void computeHash(const vector<float>& _feat, vector<int>& hash) const {
    if (_feat.size() == 0) {
      return;
    }
    hash.clear();
    Eigen::MatrixXf feat = Eigen::VectorXf::Map(&_feat[0], _feat.size());
    #if NORMALIZE_FEATS == 1
      feat = feat / feat.norm(); // normalize the feature
    #endif
    Eigen::MatrixXf res = (feat.transpose() * A - b.replicate(feat.cols(), 1)) / w;
    for (int i = 0; i < res.size(); i++) {
      hash.push_back((int) floor(res(i)));
    }
  }
  
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & w;
    ar & k;
    ar & dim;
    ar & A;
    ar & b;
  }
};

#endif

