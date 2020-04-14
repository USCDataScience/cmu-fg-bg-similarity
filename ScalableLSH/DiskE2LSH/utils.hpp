#ifndef DISKE2LSH_UTILS_HPP
#define DISKE2LSH_UTILS_HPP

#include <cmath>
#include <boost/filesystem.hpp>
#include <fstream>
#include <unordered_set>
#include "config.hpp"

namespace fs = boost::filesystem;

template<typename Dtype>
void L2Normalize(vector<Dtype>& vec) {
  Dtype norm = 0;
  for (auto el = vec.begin(); el != vec.end(); el++) {
    norm += (*el) * (*el);
  }
  norm = sqrt(norm);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = vec[i] / norm;
  }
}

template<typename Dtype>
void readList(const fs::path& fpath, vector<Dtype>& output) {
  output.clear();
  Dtype el;
  ifstream ifs(fpath.string());
  if (! ifs.is_open()) {
    cerr << "Unable to open " << fpath << endl;
    return;
  }
  while (ifs >> el) {
    output.push_back(el);
  } 
  ifs.close();
}

void getAllSearchspace(const vector<unsigned>& featcounts,
    unordered_set<long long int>& searchspace) {
  searchspace.clear();
  for (long long int i = 1; i <= featcounts.size(); i++) {
    for (long long int j = 1; j <= featcounts[i - 1]; j++) {
      searchspace.insert(i * MAXFEATPERIMG + j);
    }
  }
}

std::vector<std::string> &split(const std::string &s, 
    char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

void readResults(const fs::path& fpath,
    vector<vector<pair<float, long long>>>& allres) {
  ifstream fin(fpath.string());
  if (!fin.is_open()) {
    cerr << "Unable to read file " << fpath << endl;
    return;
  }
  string line;
  int lno = -1;
  while (getline(fin, line)) {
    lno++;
    if (line.length() <= 0) continue;
    vector<string> elems = split(line, ' ');
    for (int i = 0; i < elems.size(); i++) {
      if (elems[i].length() <= 0) continue;
      vector<string> p = split(elems[i], ':');
      allres[lno].push_back(make_pair(stof(p[1]), stoll(p[0])));
    }
  }
  fin.close();
}

/**
 * Assumes both input featid and imgid are 1 indexed
 */
long long int computeFeatId(long long int imgid, long long int featid) {
  return imgid * MAXFEATPERIMG + featid;
}

void readDupFileMatches(const fs::path& fpath,
    map<long long, vector<long long>>& matches) {
  matches.clear();
  ifstream fin(fpath.string().c_str());
  string line;
  int lno = 0;
  while (getline(fin, line)) {
    lno++;
    if (line[0] == 'U') {
      istringstream iss(line.substr(2));
      long long match;
      long long key = computeFeatId(lno, 1);
      matches[key] = vector<long long>();
      while (iss >> match) {
        matches[key].push_back(computeFeatId(match, 1));
      }
    }
  }
  fin.close();
}

vector<pair<float, long long>> augmentWithDuplicates(const fs::path& fpath,
    const vector<pair<float, long long int>>& res) {
  static bool loadedDupFile = false;
  static map<long long, vector<long long>> matches;
  if (!loadedDupFile) {
    readDupFileMatches(fpath, matches);
    loadedDupFile = true;
    cout << "Read the duplicate images file" << endl;
  }
  vector<pair<float, long long int>> finalres;
  for (int i = 0; i < res.size(); i++) {
    finalres.push_back(res[i]);
    for (int j = 0; j < matches[res[i].second].size(); j++) {
      finalres.push_back(make_pair(res[i].first, matches[res[i].second][j]));
    }
  }
  return finalres;
}

template <typename T>
vector<size_t> argsort(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
      [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

long long countNewlines(fs::path fpath) {
  ifstream fin(fpath.string());
  long long res = count(istreambuf_iterator<char>(fin), istreambuf_iterator<char>(), '\n');
  fin.close();
  return res;
}

#endif

