# FKDTree

To compile:

make

clean:
make clean

Usage: ./kdtree <option(s)> Options:
	-h,--help		Show this help message
	-n <number of points>	Specify the number of points to use for the kdtree [default 100000]
	-t 	Run the validity tests [default disabled]
	-i 	Number of iterations to run [default 1]
	-s 	Run the sequential algo
	-c 	Run the vanilla cmssw algo
	-f 	Run FKDtree algo
	-a 	Run all the algos
	-r 	Run FKDtree recursive search algo
	-b 	Run branchless FKDtree algo
	-bfs 	Run branchless FKDtree BFS algo
	-p <number of threads>	Specify the number of tbb parallel threads to use [default 1]


run memory benchmark:

./run.sh

run 

./kdtree --help

for details
