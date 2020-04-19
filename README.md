# USCDataScience: MEMEX: CMU's Foreground and Background Image Similarity Service

This is a Docker and build of an image similarity service compatible with 
[Image Space](http://github.com/nasa-jpl-memex/image_space.git) that provides
two functionalities as originally developed by [Rohit Girdrar](https://github.com/rohitgirdhar) 
from Carnegie Mellon University (CMU) in these two repos:

* [ComputeFeatures](https://github.com/rohitgirdhar/ComputeFeatures)
* [ScalableLSH](https://github.com/rohitgirdhar-cmu-experimental/ScalableLSH)

ScalableLSH contains CMU's re-implementation of ITQ and LSH. The two algorithms 
are implemented according to the following papers respectively:

* [CVPR11' Small Code](http://slazebni.cs.illinois.edu/publications/cvpr11_small_code.pdf)
* [LSH paper](http://www.mit.edu/~andoni/LSH/)

Additionally the segmentation code is based on the 
[work of Xiaolong Wang](https://github.com/xiaolonw/nips14_loc_seg_testonly/tree/master/Caffe_Segmentation) 
and the following paper:

* [Deep Joint Task Learning for Generic Object Extraction](https://papers.nips.cc/paper/5547-deep-joint-task-learning-for-generic-object-extraction). 
Proc. of Advances in Neural Information Processing Systems (NIPS), 2014

# Quick Instructions

## Build the Docker

 1. `docker build -t uscdatascience/cmu-fg-bg-similarity -f Dockerfile .`

## Run the Docker

  1. `docker run -it uscdatascience/cmu-fg-bg-similarity /bin/sh`

### In a separate window, Outside of the Docker

 1. once built, run it, get a container ID, `CID`
 2. Copy images to `/images` on the docker (e.g., from your local), `docker cp /some/path/to/imgs/local CID:/images`
 3. `docker exec -it cmu-img-sim sh`

### Inside of the Docker

 1. [/install/cmu-fg-bg-similarity/entrypoint_cmu-imgsim.sh](./entrypoint_cmu-imgsim.sh)

#### Testing a file's similarity with the rest of the corpus

 1. `cd /install/cmu-fg-bg-similarity/scripts && ./file_similarity.sh MyPic.png`

 Which should output something like:

 ```
   [
    [
        "http://localhost:8000/MyPic.png",
        1.0
    ],
    [
        "http://localhost:8000/MyPic2.png",
        0.422732
    ],
    [
        "http://localhost:8000/MyPic.jpg",
        0.349333
    ],
  ]

  ```

  Which is a list of image URLs along with the similarity to the provided image, `MyPic.png`.

#### Testing a file's foreground similarity with the rest of the corpus

 1. 1. `cd /install/cmu-fg-bg-similarity/scripts && ./file_similarity-fg.sh MyPic.png`

 You will see similar JSON output akin to the above.

#### Log file Directory

Log files are written to `/install/cmu-fg-bg-similarity/logs`.

# Questions, comments?
Send them to [Chris A. Mattmann](mailto:chris.a.mattmann@jpl.nasa.gov).

# Contributors
* Chris A. Mattmann, USC & JPL
* Tom Barber, JPL
* Rohit Girdrar, CMU
* Xiaofan Wang, CMU
* Karanjeet Singh, USC & JPL

# License
[Apache License, version 2](http://www.apache.org/licenses/LICENSE-2.0)
