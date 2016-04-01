#!/bin/bash


./kdtree -n 100000 -c -b   & ./track_memory.sh $(pidof kdtree) 0.1
