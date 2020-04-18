export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/caffe/install/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/install/zeromq410/lib
CODE_PATH=/install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/ver2
SEGSRCDIR=/install/cmu-fg-bg-similarity/segmentation/Caffe_Segmentation/segscripts
mkdir -p /install/cmu-fg-bg-similarity/pool5_fg
GLOG_logtostderr=0 ${CODE_PATH}/computeFeatures.bin \
    -i ${SEGSRCDIR}/data/final_segmentations/ \
    -q ${SEGSRCDIR}/data/ImgsList.txt \
    -n ${CODE_PATH}/../deploy_memexgpu.prototxt \
    -m /caffe/models/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -s 0\
    -t lmdb \
    -o /install/cmu-fg-bg-similarity/pool5_fg/pool5_full_normed \
    -y 1\
    -p avg \
    -z 1 \
    -f 1
