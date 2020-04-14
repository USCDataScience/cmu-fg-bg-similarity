CODE_PATH=/home/rgirdhar/data/Work/Code/0002_Retrieval/ScalableLSH/DiskE2LSH
$CODE_PATH/main.bin \
    -k 500 \
    -d /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/features/CNN_pool5_uni_normed_LMDB \
    -n /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt \
    -c /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/counts.txt \
    -o /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_bf/ \
    -m /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList.txt \
    -z
#-o /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/matches_query/ \
#-m /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/split/QueryList.txt
