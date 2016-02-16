#!/bin/bash


./kdtree -n 100000 -c -f   & ./track_memory.sh $(pidof kdtree) 0.1
