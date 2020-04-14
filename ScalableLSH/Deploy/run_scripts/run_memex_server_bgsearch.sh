BASE_PATH=/home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BASE_PATH}/external/caffe/build/lib/:/home/rgirdhar/data/Software/cpp/zeromq/install/lib/
GLOG_logtostderr=1 ../computeFeatAndSearch.bin \
    -n ../deploy_10batch.prototxt \
    -m /home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -i /home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Search/indexes/temp/bgImg_1-5M_225bit.index \
    -s /home/rgirdhar/memexdata/Dataset/processed/0001_Backpage/Features/CNN/pool5_bg_normed.old \
    -g /srv2/rgirdhar/Work/Code/0005_ObjSegment/scripts/service_scripts/temp-dir/result.jpg \
    --imgslist /home/rgirdhar/memexdata/Dataset/processed/0002_BackpageComplete/Images/lists/Images.txt \
    -p 5557  \
    --deprecated-model
