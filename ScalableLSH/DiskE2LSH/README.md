Prerequisites
-------------
- Get [LevelDB](https://github.com/google/leveldb). Compile and install.
```bash
$ cd /path/to/levedb/; make -j8
```
- Get [Boost](http://www.boost.org/). Install using
```bash
$ ./bootstrap --with-libraries=filesystem,program_options,system,serialization --exec-prefix=/path/to/install/dir
$ ./b2
$ ./b2 install
```
- Get [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). Symlink it to a folder
`Eigen` in this directory.

Set the paths to leveldb and Boost in the `Makefile` and `storage/Makefile`.

Compilation
-----------
```bash
$ make
$ cd storage; make
```

Usage Instructions
------------------

1. Compute a leveldb index of the features.
  - Use `storage/main.cpp`
2. Use `main.cpp` for the hashing/searching now. A sample command to run hashing and then
search would be:
```bash
$ ./main.bin -d storage/selsearch_feats_normalized3 -n 641581 -q storage/marked_feats_normalized/ -m 237 -s full.model
```

Notes
-----
- Ranking ~3M features on HDD with LMDB index takes ~20sec.
- The feature index is `imid * 10K + featid`. imid and featid here are 0 indexed. This feature index is then incremented by 1 while storing

