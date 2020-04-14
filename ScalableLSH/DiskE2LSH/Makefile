#export PATH=/opt/centos/devtoolset-1.0/root/usr/bin/:$PATH
#export LD_LIBRARY_PATH=/srv2/rgirdhar/Software/cpp/zeromq/install/lib:$LD_LIBRARY_PATH
CPPFLAGS += -std=c++0x -fopenmp -g -O2 -llmdb
LDFLAGS += -lleveldb -lboost_serialization -lboost_system -lboost_program_options -lboost_filesystem -lglog -lz -lzmq -llmdb
LIBS += -L/install/zeromq410/lib/ -L/install/boost157/lib/
INC += -IEigen/  -I/install/zeromq410/cppzmq/ -I /install/zeromq410/include/ -I/install/boost157/ -I/install/boost_1_57_0/




all: buildIndex.bin main.bin computePairwiseSim.bin

%.bin: %.cpp storage/DiskVector.hpp LSHFunc_ITQ.hpp Table.hpp Resorter.hpp LSH.hpp config.hpp storage/DiskVectorLMDB.hpp
	$(CXX) $(CPPFLAGS) $(INC) $< -o $@ $(LIBS) $(LDFLAGS)

clean:
	rm *.bin

