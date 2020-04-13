export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/caffe/install/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/install/zeromq410/lib
CODE_PATH=/install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/ver2
GLOG_logtostderr=0 ${CODE_PATH}/computeFeatures.bin \
    -i /ctceu \
    -q ${CODE_PATH}/fileList.txt \
    -n ${CODE_PATH}/../deploy_memexgpu.prototxt \
    -m /caffe/models/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -s 0\
    -t lmdb \
    -o /install/cmu-fg-bg-similarity/pool5/pool5_full_normed \
    -y 1\
    -p avg \
    -z 1 \
    -f 1
