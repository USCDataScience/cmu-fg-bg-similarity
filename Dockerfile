FROM centos:7

RUN yum update -y && yum install -y epel-release
RUN rpm -v --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
RUN rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
RUN yum install -y http://rpms.remirepo.net/enterprise/remi-release-7.rpm
RUN rpm -Uvh http://repo.webtatic.com/yum/el6/latest.rpm
RUN yum install -y bc zlib-devel atlas-devel bzip2-devel gcc-c++ ffmpeg ffmpeg-devel make pkgconfig \
                   gtk2-devel perl cmake cmake3 git libcurl-devel.x86_64 unzip wget \
                   protobuf-devel lapack-devel leveldb-devel snappy-devel opencv-devel hdf5-devel \
		   libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel libtiff-devel libwebp-devel \
                   libdc1394-devel libv4l-devel gstreamer-plugins-base-devel \
		   boost-devel lmdb-devel openblas-devel centos-release-scl devtoolset-7

RUN yum groupinstall -y 'Development Tools'
WORKDIR /install
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
RUN bash Miniconda2-latest-Linux-x86_64.sh -b -p /install/miniconda
RUN echo "export PATH=/install/miniconda/bin:$PATH" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"

ENV PATH /install/miniconda/bin:${PATH}
RUN conda install numpy PyYAML

WORKDIR /install
RUN wget http://sourceforge.net/projects/boost/files/boost/1.57.0/boost_1_57_0.tar.gz
RUN tar -xzvf boost_1_57_0.tar.gz
WORKDIR /install/boost_1_57_0
RUN /install/boost_1_57_0/bootstrap.sh --prefix=/install/boost157
RUN /install/boost_1_57_0/b2 install


WORKDIR /install
RUN wget https://github.com/opencv/opencv/archive/3.1.0.zip
RUN unzip 3.1.0.zip
#RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.1.0.zip
#RUN unzip opencv_contrib.zip
WORKDIR /install/opencv-3.1.0
RUN mkdir build
WORKDIR /install/opencv-3.1.0/build
#          -DOPENCV_EXTRA_MODULES_PATH:PATH=../../opencv_contrib-3.1.0/modules \
RUN cmake -DCMAKE_BUILD_TYPE=Release \
	  -DBUILD_PYTHON_SUPPORT:BOOL=ON \
	  -DBUILD_EXAMPLES:BOOL=ON \
	  -DPYTHON_DEFAULT_EXECUTABLE:PATH=/install/miniconda/bin/python \
	  -DPYTHON_INCLUDE_DIRS:PATH=/install/miniconda/include \
	  -DPYTHON_EXECUTABLE:PATH=/install/miniconda/bin/python \
	  -DPYTHON_LIBRARY:PATH=/install/miniconda/lib/libpython2.7.so.1.0 \
	  -DBUILD_opencv_python3:BOOL=OFF \
	  -DBUILD_opencv_python2:BOOL=ON \
	  -DWITH_IPP:BOOL=OFF \
	  -DWITH_FFMPEG:BOOL=ON \
	  -DWITH_V4L:BOOL=ON .. \
    	  -DOPENCV_BUILD_3RDPARTY_LIBS:BOOL=ON \
          -DBUILD_PNG:BOOL=ON \
	  -DWITH_PNG:BOOL=ON \
	  -DBUILD_JPEG:BOOL=ON \
	  -DWITH_JPEG:BOOL=ON \
	  -DBUILD_TIFF=ON \
	  -DWITH_TIFF=ON \
	  -DINSTALL_PYTHON_EXAMPLES:BOOL=ON \
	  -DOPENCV_GENERATE_PKGCONFIG:BOOL=ON \
	  -DCMAKE_INSTALL_PREFIX:PATH=/install/cv310 ..
RUN make -j7 && make install

WORKDIR /install
RUN wget https://archive.org/download/zeromq_4.1.0/zeromq-4.1.0-rc1.tar.gz
RUN tar -xzvf zeromq-4.1.0-rc1.tar.gz
WORKDIR /install/zeromq-4.1.0
RUN ./configure --prefix=/install/zeromq410
RUN make
RUN make install
WORKDIR /install/zeromq-4.1.0

ENV CMAKE_PREFIX_PATH "/install/zeromq410:$CMAKE_PREFIX_PATH"
ENV PKG_CONFIG_PATH "/install/zeromq410/lib/pkgconfig:$PKG_CONFIG_PATH"
RUN wget https://github.com/zeromq/cppzmq/archive/v4.2.2.zip
RUN unzip v4.2.2.zip
WORKDIR /install/zeromq-4.1.0/cppzmq-4.2.2
RUN mkdir build
WORKDIR /install/zeromq-4.1.0/cppzmq-4.2.2/build
RUN cmake3 ..
RUN make -j4 install

WORKDIR /install
RUN wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/google-glog/glog-0.3.3.tar.gz
RUN tar -xzvf glog-0.3.3.tar.gz
WORKDIR /install/glog-0.3.3
RUN ./configure
RUN make && make install

WORKDIR /install
RUN wget https://github.com/schuhschuh/gflags/archive/v2.2.1.zip
RUN unzip v2.2.1.zip
WORKDIR /install/gflags-2.2.1
RUN mkdir build
WORKDIR /install/gflags-2.2.1/build
ENV CXXFLAGS "-fPIC"
RUN cmake ..
RUN make VERBOSE=1 && make && make install

RUN mkdir /caffe /caffe/models /caffe/build \
 && curl -L https://github.com/BVLC/caffe/archive/e79bc8f.tar.gz >caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a.tar.gz \
 && tar -xzf caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a.tar.gz \
 && mv caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a /caffe/source \
 && rm caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a.tar.gz

# - Fetching data and model files
RUN /caffe/source/data/ilsvrc12/get_ilsvrc_aux.sh \
 && /caffe/source/scripts/download_model_binary.py /caffe/source/models/bvlc_alexnet \
 && /caffe/source/scripts/download_model_binary.py /caffe/source/models/bvlc_reference_caffenet \
 && mv /caffe/source/data/ilsvrc12/imagenet_mean.binaryproto /caffe/models/ \
 && mv /caffe/source/models/bvlc_alexnet/bvlc_alexnet.caffemodel /caffe/models/ \
 && mv /caffe/source/models/bvlc_alexnet/deploy.prototxt /caffe/models/ \
 && mv /caffe/source/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel /caffe/models/

# - Build, linking to deps
RUN cd /caffe/build \
 && cmake \
    -DBLAS:STRING=open \ 
    -DBLAS_LIB:PATH=/usr/lib64/libblas.so \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCPU_ONLY:BOOL=ON \
    -DPYTHON_EXECUTABLE:PATH=/install/miniconda/bin/python2.7 \
    -DPYTHON_INCLUDE_DIR:PATH=/install/miniconda/include/python2.7 \
    -DPYTHON_INCLUDE_DIR2:PATH=/install/miniconda/include/python2.7 \
    -DPYTHON_LIBRARY:PATH=/install/miniconda/lib/libpython2.7.so \
    -DUSE_CUDNN:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=/caffe/install \
    /caffe/source \
 && make install -j12 \
 && cd \
 && rm -r /caffe/source /caffe/build
ENV PATH="/caffe/install/bin:${PATH}" \
    PYTHONPATH="/caffe/install/python:${PYTHONPATH}"

# Install CMU FG/BG software
WORKDIR /install
RUN git clone http://github.com/USCDataScience/cmu-fg-bg-similarity.git
WORKDIR /install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/ver2/
RUN make
ENV LD_LIBRARY_PATH /caffe/install/lib:/usr/local/lib:/install/boost157/lib:/install/cv310/lib:/install/zeromq410/lib:
