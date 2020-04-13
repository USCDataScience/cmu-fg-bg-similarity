#ifndef SLIDINGWINDOWCONFIG_HPP
#define SLIDINGWINDOWCONFIG_HPP

// for sliding window
#define SLIDINGWIN_MIN_STRIDE 24
#define SLIDINGWIN_MIN_SZ_X 32
#define SLIDINGWIN_MIN_SZ_Y 32
#define SLIDINGWIN_WINDOW_RATIO 0.25f // perc of the dimensions
#define SLIDINGWIN_STRIDE_RATIO 0.5f // perc of window size

// for pruning using background
#define PERC_FGOVERLAP_FOR_BG 0.5f // less than this overlap means bg

#endif
