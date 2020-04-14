#ifndef TABLE_HPP
#define TABLE_HPP

// This is needed so that I can create a hash func for
// dynamic_bitset. It makes m_bits publically accessible.
// ref: http://stackoverflow.com/a/3897217/1558890
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include "LSHFunc_ITQ.hpp"
#include "utils.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp> 
#include <boost/serialization/vector.hpp> 
#include <boost/serialization/unordered_set.hpp>
#include <boost/functional/hash.hpp>
#include <boost/dynamic_bitset.hpp>

struct dynbitset_hash {
  template <typename B, typename A>
    std::size_t operator()(const boost::dynamic_bitset<B, A>& bs) const {
      return boost::hash_value(bs.m_bits);
    }
};

class Table {
  friend class boost::serialization::access;
  LSHFunc_ITQ lshFunc;
  unordered_map<boost::dynamic_bitset<>, unordered_set<long long>, dynbitset_hash> index;
  vector<boost::dynamic_bitset<>> indexKeys; // maintain keys to above map, for fast hamming search
public:
  Table(int k) : lshFunc(k) {}
  Table() {} // used for serializing
  template <typename T>
  void train(const vector<vector<T>>& sampleData) {
    lshFunc.train(sampleData);
  }
  void insert(const vector<float>& feat, long long int label) {
    boost::dynamic_bitset<> hash = lshFunc.computeHash(feat);

    auto pos = index.find(hash);
    if (pos == index.end()) {
      unordered_set<long long int> lst; 
      lst.insert(label);
      index[hash] = lst;
      indexKeys.push_back(hash);
    } else {
      pos->second.insert(label);
    }
  }
  bool search_exact(const vector<float>& feat, unordered_set<long long int>& output) const {
    boost::dynamic_bitset<> hash = lshFunc.computeHash(feat);
    auto pos = index.find(hash);
    if (pos != index.end()) {
      output = pos->second;
      return true;
    }
    return false;
  }
  /**
   * Hamming distance search
   */
  bool search(const vector<float>& feat, unordered_set<long long>& output, int nRerank) const {
    output.clear();
    vector<int> hamdists(indexKeys.size(), 0);
    boost::dynamic_bitset<> hash = lshFunc.computeHash(feat);
    #pragma omp parallel for
    for (size_t i = 0; i < indexKeys.size(); i++) {
      hamdists[i] = (indexKeys[i] ^ hash).count();
    }
    vector<size_t> order = argsort(hamdists);
    for (size_t i = 0; i < min(hamdists.size(), (size_t) nRerank); i++) {
      // ref: http://stackoverflow.com/a/262872
      unordered_set<long long> match = index.at(indexKeys[order[i]]);
      output.insert(match.begin(), match.end());
    }
  }
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & lshFunc;
    ar & index;
    ar & indexKeys;
  }
};

/**
 * The following code is to implement serialization for 
 * boost::dynamic_bitset. Implementation directly copied from
 * http://stackoverflow.com/a/31014564
 */
namespace boost { namespace serialization {
  template <typename Ar, typename Block, typename Alloc>
    void save(Ar& ar, dynamic_bitset<Block, Alloc> const& bs, unsigned) {
      size_t num_bits = bs.size();
      std::vector<Block> blocks(bs.num_blocks());
      to_block_range(bs, blocks.begin());

      ar & num_bits & blocks;
    }

  template <typename Ar, typename Block, typename Alloc>
    void load(Ar& ar, dynamic_bitset<Block, Alloc>& bs, unsigned) {
      size_t num_bits;
      std::vector<Block> blocks;
      ar & num_bits & blocks;

      bs.resize(num_bits);
      from_block_range(blocks.begin(), blocks.end(), bs);
      bs.resize(num_bits);
    }

  template <typename Ar, typename Block, typename Alloc>
    void serialize(Ar& ar, dynamic_bitset<Block, Alloc>& bs, unsigned version) {
      split_free(ar, bs, version);
    }
} }

#endif

