export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/install/zeromq410/lib
CODE_PATH=/install/cmu-fg-bg-similarity/ScalableLSH/DiskE2LSH
SEGSRCDIR=/install/cmu-fg-bg-similarity/segmentation/Caffe_Segmentation/segscripts
mkdir -p /install/cmu-fg-bg-similarity/indexes
GLOG_logtostderr=0 nice -n 19 $CODE_PATH/buildIndex.bin \
    -d /install/cmu-fg-bg-similarity/pool5_fg/pool5_full_normed \
    -n ${SEGSRCDIR}/data/ImgsList.txt \
    -s /install/cmu-fg-bg-similarity/indexes/FgImg_ITQ_256bit.index \
    -b 256 \
    -t 1 \
    -a 900 \
    --nTrain 50000
