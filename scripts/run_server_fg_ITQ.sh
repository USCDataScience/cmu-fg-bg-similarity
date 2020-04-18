mkdir -p /install/cmu-fg-bg-similarity/logs
BASE_PATH=/install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/
CODE_PATH=/install/cmu-fg-bg-similarity/ScalableLSH/Deploy/
SEGSRCDIR=/install/cmu-fg-bg-similarity/segmentation/Caffe_Segmentation/segscripts
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/caffe/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/install/zeromq410/lib
GLOG_logtostderr=1 ${CODE_PATH}/computeFeatAndSearch.bin \
    -n ${CODE_PATH}/deploy.prototxt \
    -m /caffe/models/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -i /install/cmu-fg-bg-similarity/indexes/FgImg_ITQ_256bit.index \
    -s /install/cmu-fg-bg-similarity/pool5_fg/pool5_full_normed \
    --imgslist ${SEGSRCDIR}/data/ImgsList.txt  \
    --nPathParts -1 \
    --port-num 5569 \
    --num-output 100 \
    --nRerank 1000\
    -f 1  > /install/cmu-fg-bg-similarity/logs/itq-fg-server.log 2>&1&

