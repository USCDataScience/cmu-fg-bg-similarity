# Segent the following:
CAFFE_ROOT := /caffe/install
FEAT_EXT_CODE_ROOT := /install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/ver2
#CUDA_ROOT := /usr/local/cuda
CPU_ONLY := 1

# for the zmq
INC += -I$(CAFFE_ROOT)/include/ -I/caffe/source/include/ -I/install/cv310/include/ -I../ -I/install/zeromq410/cppzmq/ -I /install/zeromq410/include/ -I/install/boost157/ -I/install/boost_1_57_0/ -I/usr/include/
LIBS += -L/install/zeromq410/lib/ -L$(CAFFE_ROOT)/lib/ -L/install/cv310/lib/ -L/install/boost157/lib/
LDFLAGS += -lzmq -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
CXXFLAGS += -std=c++0x -fopenmp -O2 -g -DUSE_OPENCV
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -lboost_thread -llmdb -lz -lcurl

# for the feature extraction utils.hpp
INC += -I$(FEAT_EXT_CODE_ROOT)

# for the search code
SEARCH_CODE_PATH=../DiskE2LSH
INC += -I$(SEARCH_CODE_PATH) -I$(SEARCH_CODE_PATH)/Eigen



ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CXXFLAGS += -DCPU_ONLY
endif

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile
	$(CXX)  \
		$(CXXFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) 

clean:
	rm -rf *.bin
