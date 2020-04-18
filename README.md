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

# Quick Instructions

## Build the Docker

 1. `docker build -t cmu-img-sim:latest -f Dockerfile .`

## Run the Docker

  1. `docker run -it cmu-img-sim /bin/sh`

### In a separate window, Outside of the Docker

 1. once built, run it, get a container ID, `CID`
 2. Copy images to `/ctceu` on the docker (e.g., from your local), `docker cp /some/path/to/imgs/local CID:/ctceu`
 3. `docker exec -it cmu-img-sim sh`

### Inside of the Docker

 1. `/install/cmu-fg-bg-similarity/scripts/clean-paths.sh`
 2. `/install/cmu-fg-bg-similarity/scripts/gen-file-list.sh`
 3. `cd /install/cmu-fg-bg-similarity/scripts/ && ./run_FeatExt_full.sh`
 4. `cd /install/cmu-fg-bg-similarity/scripts/ && ./buildIndex_ITQ.sh`
 5. `cd /install/cmu-fg-bg-similarity/scripts  && ./run_server_ITQ.sh`
 6. `cd /install/cmu-fg-bg-similarity/scripts && ./web_services.sh`

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
