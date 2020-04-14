#ifndef RESORTER_HPP
#define RESORTER_HPP

#include "storage/DiskVectorLMDB.hpp"
#include "storage/DiskVector.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#define MAX_RESORT_BATCH_SIZE 10000 // resort these many at a time
// Similarity Metrics
#define SIM_METRIC_COSINE 1
#define SIM_METRIC_EUCLIDEAN 2

class Resorter {
public:
  float static computeDot(vector<float> a, vector<float> b) {
    float ans = 0;
    for (auto it = a.begin(), it2 = b.begin(); 
        it != a.end() && it2 != b.end(); it++, it2++) {
      ans += (*it) * (*it2);
    }
    return ans;
  }
  
  /**
   * When one DiskVector has all the features
   */
  void static resort(const unordered_set<long long int>& matches, 
      const std::shared_ptr<DiskVector<vector<float>>>& feats,
      vector<float>& qfeat,
      vector<pair<float, long long int>>& res) {
    res.clear();
    if (matches.size() == 0) {
      cerr << "0 matches input to resorter..";
      return;
    }
    #if NORMALIZE_FEATS == 1
      L2Normalize(qfeat); // no longer required as storing normalized
    #endif
    Eigen::MatrixXf qfeat_mat = Eigen::VectorXf::Map(&qfeat[0], qfeat.size());
    
    // Batch process the scoring
    int nMatches = matches.size();
    int nBatches = ceil(nMatches * 1.0f / MAX_RESORT_BATCH_SIZE);
    auto match = matches.begin();
    vector<vector<float>> feats_vec;
    vector<pair<float, long long int>> res_batch;
   
    for (int batch = 0; batch < nBatches; batch++) {
      feats_vec.clear();
      res_batch.clear();
      int batchSize = 0;
      for (; match != matches.end() && batchSize < MAX_RESORT_BATCH_SIZE;
          match++, batchSize++) {
        // TODO: Avoid this, use pre-alloc of memory
        vector<float> temp;
        feats->Get(*match, temp);
        #if NORMALIZE_FEATS == 1
          L2Normalize(temp);
        #endif
        feats_vec.push_back(temp);
        // for output
        res_batch.push_back(make_pair(0.0f, *match));
      }
      cout << "read all"; cout.flush();
      Eigen::MatrixXf feats_mat(matches.size(), qfeat.size());
      for (int i = 0; i < feats_vec.size(); i++) {
        feats_mat.row(i) = Eigen::VectorXf::Map(&feats_vec[i][0], feats_vec[i].size());
      }
      Eigen::MatrixXf cos_scores = qfeat_mat.transpose() * feats_mat.transpose();
      cout << "rescored"; cout.flush();

      for (int i = 0; i < res_batch.size(); i++) {
        res_batch[i].first = cos_scores(i);
      }
      res.insert(res.end(), res_batch.begin(), res_batch.end());
    }
    sort(res.begin(), res.end());
    reverse(res.begin(), res.end());
  }

  /**
   * Multicore reranking
   */
  void static resort_multicore(const unordered_set<long long int>& matches, 
      const std::shared_ptr<DiskVectorLMDB<vector<float>>>& feats,
      vector<float>& qfeat,
      vector<pair<float, long long int>>& res,
      int simMetric = SIM_METRIC_COSINE) {
    res.clear();
    vector<long long int> matches_vec(matches.begin(), matches.end());
    if (matches.size() == 0) {
      cerr << "0 matches input to resorter..";
      return;
    }

    #if NORMALIZE_FEATS == 1
      L2Normalize(qfeat);
    #endif
    Eigen::MatrixXf qfeat_mat = Eigen::VectorXf::Map(&qfeat[0], qfeat.size());
    qfeat_mat.normalize();
    
    int nMatches = matches.size();
    vector<float> scores(nMatches, 0);

    #pragma omp parallel for shared(scores) schedule(dynamic, 1)
    for (int i = 0; i < nMatches; i++) {
      vector<float> temp;
      if (!feats->Get(matches_vec[i], temp)) {
        scores[i] = 0;
        continue;
      }
      #if NORMALIZE_FEATS == 1
        L2Normalize(temp);
      #endif
      Eigen::MatrixXf match_mat = Eigen::VectorXf::Map(&temp[0], temp.size());
      if (simMetric == SIM_METRIC_COSINE) {
        Eigen::MatrixXf sim = qfeat_mat.transpose() * match_mat;
        scores[i] = sim(0); 
      } else if (simMetric == SIM_METRIC_EUCLIDEAN) {
        match_mat.normalize();
        // since I need similarity, and 2 is the max distance between
        // any 2 unit vectors under any metric distance (triangle ineq)
        scores[i] = 2.0f - (qfeat_mat - match_mat).norm();
        if (scores[i] < 0) {
          
        }
      } else {
        cerr << "Distance metric not implemented!" << endl;
        scores[i] = -1;
      }
    }

    for (int i = 0; i < nMatches; i++) {
      res.push_back(make_pair(scores[i], matches_vec[i]));
    }

    sort(res.begin(), res.end());
    reverse(res.begin(), res.end());
  }

  /**
   * A one-off function to do nxn similarity for all features in an image
   */
  void static computePairwiseSim(
      const std::shared_ptr<DiskVectorLMDB<vector<float>>>& feats,
      int imgid, // 1-indexed
      int numFeat, // num of features in this image
      Eigen::MatrixXf& simsOutput) {
    vector<vector<float>> data;
    for (int i = 1; i <= numFeat; i++) {
      int featid = computeFeatId(imgid, i);
      vector<float> temp;
      if (!feats->Get(featid, temp)) {
        cerr << "Couldn't read featid = " << featid << endl;
        continue;
      }
      data.push_back(temp);
    }
    if (data.size() == 0) return;
    Eigen::MatrixXf featmat(numFeat, data[0].size());
    for (int i =0; i < numFeat; i++) {
      featmat.row(i) = Eigen::VectorXf::Map(&data[i][0], data[i].size());
    }
    simsOutput = featmat * featmat.transpose();
  }

  /**
   * When multiple DiskVector have the features
   */
  // NOT BEING USED currently. TODO: remove it
  void static resort2(const vector<pair<int, int>>& matches, 
      const vector<boost::shared_ptr<DiskVector<vector<float>>>>& featstor,
      const vector<float>& qfeat,
      vector<pair<float, pair<int, int>>>& res) {
    res.clear();
    Eigen::MatrixXf qfeat_mat = Eigen::VectorXf::Map(&qfeat[0], qfeat.size());
    
    // Batch process the scoring
    int nMatches = matches.size();
    int nBatches = ceil(nMatches * 1.0f / MAX_RESORT_BATCH_SIZE);
    auto match = matches.begin();
    vector<vector<float>> feats_vec;
    vector<pair<float, pair<int, int>>> res_batch;
    
    for (int batch = 0; batch < nBatches; batch++) {
      feats_vec.clear();
      res_batch.clear();
      int batchSize = 0;
      for (; match != matches.end() && batchSize < MAX_RESORT_BATCH_SIZE;
          match++, batchSize++) {
        // TODO: Avoid this, use pre-alloc of memory
        vector<float> temp;
        featstor[match->first]->Get(match->second, temp);

        L2Normalize(temp);
        feats_vec.push_back(temp);
        // for output
        res_batch.push_back(make_pair(0.0f, *match));
      }
      Eigen::MatrixXf feats_mat(matches.size(), qfeat.size());
      for (int i = 0; i < feats_vec.size(); i++) {
        feats_mat.row(i) = Eigen::VectorXf::Map(&feats_vec[i][0], feats_vec[i].size());
      }
      Eigen::MatrixXf cos_scores = qfeat_mat.transpose() * feats_mat.transpose();

      for (int i = 0; i < res_batch.size(); i++) {
        res_batch[i].first = cos_scores(i);
      }
      res.insert(res.end(), res_batch.begin(), res_batch.end());
    }
    sort(res.begin(), res.end());
    reverse(res.begin(), res.end());
  }

};

#endif
