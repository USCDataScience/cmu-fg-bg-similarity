CODE_PATH=/home/rgirdhar/data/Work/Code/0001_FeatureExtraction/ComputeFeatures/Features/CNN
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CODE_PATH}/external/caffe_dev_MemLayerWithMat/build/lib/
${CODE_PATH}/computeFeatures.bin \
    -i /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/corpus/ \
    -q /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/ImgsList.txt \
    -n ${CODE_PATH}/deploy_memexgpu.prototxt \
    -m /home/rgirdhar/data/Software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    -l pool5 \
    -o /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_pool5_uni_normed_LMDB \
    -w /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/selsearch_boxes \
    -y # normalize the features
