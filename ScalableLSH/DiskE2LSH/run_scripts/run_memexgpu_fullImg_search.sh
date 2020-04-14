CODE_PATH=/home/rgirdhar/data/Work/Code/0002_Retrieval/ScalableLSH/DiskE2LSH
$CODE_PATH/main.bin \
    -k 500 \
    -d /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/features/CNN_pool5_fullImg_normed_LMDB \
    -n /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt \
    -l /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/search_models/search_fullImg_20bit.index \
    -o /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/ \
    -m /home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList.txt
#-o /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/matches_query/ \
#-m /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/split/QueryList.txt
