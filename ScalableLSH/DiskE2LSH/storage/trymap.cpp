#include <iostream>
#include "DiskMapLMDB.hpp"
#include <vector>

using namespace std;

int main() {
  DiskMapLMDB<vector<int>, vector<int>> mp("test.index");
  vector<int> a(10, 1);
  vector<int> b(10, 2);
  mp.Put(a, b);
  vector<int> c(10, 1);
  vector<int> res;
  mp.Get(c, res);
  for (int i = 0; i < 10; i++) {
    cout << res[i] << " ";
  }
  cout << endl;
  mp.Put(b, a);
  mp.Get(c, res);
  for (int i = 0; i < 10; i++) {
    cout << res[i] << " ";
  }
  cout << endl;
  mp.Get(b, res);
  for (int i = 0; i < 10; i++) {
    cout << res[i] << " ";
  }
  cout << endl;

  return 0;
}

