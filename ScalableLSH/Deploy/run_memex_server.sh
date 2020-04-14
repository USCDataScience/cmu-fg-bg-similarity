BASE_PATH=/home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BASE_PATH}/external/caffe_dev_MemLayerWithMat/build/lib/:/home/rgirdhar/data/Software/cpp/zeromq/install/lib/
GLOG_logtostderr=1 ./computeFeatAndSearch.bin \
    -n deploy.prototxt \
    -m /home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -i /home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Search/indexes/fullImg_250bit.index \
    -s /home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Features/CNN/pool5_normed \
    --imgslist /home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Images/lists/Images.txt

