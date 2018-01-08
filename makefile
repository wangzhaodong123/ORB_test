INCLUDE_OPENCV = $(shell pkg-config --cflags opencv)
LIBDIR_OPENCV = $(shell pkg-config --libs opencv)

all:orb_test

orb_test:orb_test.o
	g++ -std=c++11 -o orb_test orb_test.o $(LIBDIR_OPENCV) -L /usr/local/lib/libopencv_nonfree.so.2.4
orb_test.o:orb_test.cpp
	g++ -std=c++11 -c orb_test.cpp $(INCLUDE_IPENCV)
clean:
	rm -f *.o orb_test
