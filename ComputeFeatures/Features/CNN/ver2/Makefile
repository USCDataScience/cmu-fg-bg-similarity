# Set the following:
#CAFFE_INSTALL_DIR = /IUS/homes4/rohytg/software/caffe_install
#CAFFE_ROOT := ../external/caffe
CAFFE_ROOT := /caffe/install
CUDA_ROOT := /usr/local/cuda
#INC += -I$(CAFFE_INSTALL_DIR)/gflags_install/include -I$(CAFFE_INSTALL_DIR)/glog_install/include -I$(CAFFE_INSTALL_DIR)/mdb_install/include -I ~/software/boost/include
#LIBS += -L$(CAFFE_INSTALL_DIR)/gflags_install/lib -L$(CAFFE_INSTALL_DIR)/glog_install/lib -L$(CAFFE_INSTALL_DIR)/mdb_install/lib -L ~/software/boost/libs
CPU_ONLY := 1

CXXFLAGS += -std=c++0x -fopenmp -O2 -g
#LIBS += -L/root/tools/anaconda/lib -L$(CAFFE_ROOT)/build/lib -L/root/tools/zeromq/lib/ -L/root/tools/opencv/lib/ -L/root/tools/boost/lib/
LIBS += -L/install/miniconda/lib -L$(CAFFE_ROOT)/lib -L/install/zeromq410/lib/ -L/install/cv310/lib/ -L/install/cv310/share/OpenCV/3rdparty/lib/ -L/install/boost157/lib/
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -llmdb `pkg-config --libs /install/cv310/lib/pkgconfig/opencv.pc` -lhdf5 -lzmq
INC += -I$(CAFFE_ROOT)/include/ -I/caffe/source/include/ -I/install/cv310/include/ -I../ -I/install/zeromq410/cppzmq/ -I /install/zeromq410/include/ -I/install/boost157/ -I/install/boost_1_57_0/
ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CXXFLAGS += -DCPU_ONLY
endif
CXXFLAGS += -DUSE_OPENCV

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile utils.hpp ../external/DiskVector/DiskVectorLMDB.hpp Config.hpp
	$(CXX)  \
		$(CXXFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) 

clean:
	rm -rf *.bin
