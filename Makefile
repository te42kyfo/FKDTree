CFLAGS=-std=c++11 -ftree-vectorize -march=native -ffast-math -Ofast -march=native -mtune=native

CC=g++

all: kdtree

kdtree: main.cpp
	$(CC) $(CFLAGS) main.cpp -o kdtree

clean:
	rm kdtree mem.log*
