CPPFLAGS += -std=c++0x -g
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lglog -llmdb -lboost_filesystem -lboost_program_options -lz
LIBS += -L ~/software/leveldb/ -L /mnt/data/Softwares/CPP/boost/boost_1_57_0/stage/lib/ -L/mnt/data/Softwares/CPP/zlib/install/lib/ -L/mnt/data/Softwares/CPP/lmdb/libraries/liblmdb/
INC += -I ~/software/leveldb/include/ -I /mnt/data/Softwares/CPP/boost/boost_1_57_0/ -I/mnt/data/Softwares/CPP/zlib/install/include/ -I/mnt/data/Softwares/CPP/lmdb/libraries/liblmdb/

all: main.bin read.bin trymap.bin tools/compressStore.bin

%.bin: %.cpp DiskVector.hpp DiskVectorLMDB.hpp
	$(CXX) \
		$(CPPFLAGS) \
		$(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2

