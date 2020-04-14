
Usage
-----
Tested for use to store `vector<float>` on disk. Simply refer to `main.cpp`
and `read.cpp` to quickly get started.

Updates
-------
- Moved to [LMDB](http://symas.com/mdb/) as the datastore backend
  (`DiskVectorLMDB` class)
  - Primarily because it is optimized for reads as compared to writes,
  and this package is developed to write the data once and then read multiple
  times. (A complete benchmark
  comparing LMDB, leveldb etc is [here](http://symas.com/mdb/microbench/)).
  - Code using the `leveldb` backend is still available through the `DiskVector` class.

Python Wrapper
--------------

`DiskVectorLMDB` now supports a python wrapper using Boost.Python.

### Compile
In `$ROOT_DIR/python`, set path to boost library in `Makefile`, and then `make`


### Usage (in python)

The basic API is exactly same as C++, so refer to the C++ code for more info

```python
>>> import PyDiskVectorLMDB
>>> stor = PyDiskVectorLMDB.DiskVectorLMDB('temp', False) # 2nd arg is readonly=T/F
>>> # generate a feature vector to be inserted
>>> f = PyDiskVectorLMDB.FeatureVector()
>>> f.append(1)
>>> f.append(2)
>>> # insert
>>> stor.Put(1, f)
>>> # retrieve
>>> q = PyDiskVectorLMDB.FeatureVector()
>>> stor.Get(1, q)
>>> # Print
>>> for i in q:
...   print i
```

#### Note
- Currently, only supports a list of floating values to be inserted/retrieved
from the `DiskVector`. This is the `FeatureVector` datatype used above.

Note
-----

- `DiskVectorLMDB` allows for update to the model. It does not support duplicate
keys, so, if you add another value with a key that already existed, it'll simply
overwrite it (**without warning**).
Also, it will not store the old values of the features, so no extra space will be used.
