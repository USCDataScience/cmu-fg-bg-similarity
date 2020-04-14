#ifndef CONFIG_HPP
#define CONFIG_HPP

#define NORMALIZE_FEATS 1 // no need since already storing normalized 
//#define RAND_SAMPLE 1000 // set # if you want output for a # randsample of the features
                         // comment out (not define) if you want all patches
#define MAXFEATPERIMG 10000 // the max number of selsearch features per image. Used to generate the feature idx

#endif
