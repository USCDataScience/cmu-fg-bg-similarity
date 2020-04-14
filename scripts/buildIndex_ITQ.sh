export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/install/zeromq410/lib
CODE_PATH=/install/cmu-fg-bg-similarity/ScalableLSH/DiskE2LSH
mkdir -p /install/cmu-fg-bg-similarity/indexes
GLOG_logtostderr=0 nice -n 19 $CODE_PATH/buildIndex.bin \
    -d /install/cmu-fg-bg-similarity/pool5/pool5_full_normed \
    -n /install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/ver2/fileList.txt  \
    -s /install/cmu-fg-bg-similarity/indexes/FullImg_ITQ_256bit.index \
    -b 256 \
    -t 1 \
    -a 900 \
    --nTrain 50000
