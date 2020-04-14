CODE_PATH=/home/rgirdhar/data/Work/Code/0002_Retrieval/ScalableLSH/DiskE2LSH
$CODE_PATH/main.bin \
    -d /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/features/CNN_pool5_uni_normed_LMDB \
    -n /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt \
    -c /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/counts.txt \
    -s /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/search_models/search_250bit.index
