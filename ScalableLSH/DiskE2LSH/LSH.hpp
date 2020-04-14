#ifndef LSH_HPP
#define LSH_HPP

#include "Table.hpp"
#include <boost/serialization/serialization.hpp>

class LSH {
  friend class boost::serialization::access;
  vector<Table> tables;
public:
  long long lastLabelInserted; // IMPORTANT. It only inserts a feature if its index is greater than
                               // this number. So make sure the labels are increasing order
  LSH(int k, int L) {
    for (int i = 0; i < L; i++) {
      tables.push_back(Table(k));
    }
    lastLabelInserted = -1;
  }
  template <typename T>
  void train(const vector<vector<T>>& sampleData) {
    if (tables.size() > 1) {
      // TODO (rgirdhar): Allow multiple tables to be trained using 
      // different randomly selected sets of data. Until then,
      // just use one table
      cerr << "WARNING:: Training multiple tables with same data"
           << endl;
    }
    for (int i = 0; i < tables.size(); i++) {
      tables[i].train(sampleData);
    }
  }
  void insert(const vector<float>& feat, long long int label) {
    lastLabelInserted = label;
    #pragma omp parallel for
    for (int i = 0; i < tables.size(); i++) {
      tables[i].insert(feat, label);
    }
  }
  void search(const vector<float>& feat, 
      unordered_set<long long int>& output,
      int nRerank) const {
    output.clear();
//    #pragma omp parallel for
    for (int i = 0; i < tables.size(); i++) {
      unordered_set<long long int> part;
      tables[i].search(feat, part, nRerank);
//      #pragma omp critical
      output.insert(part.begin(), part.end());
    }
  }
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & tables; 
    ar & lastLabelInserted;
  }
};

#endif
