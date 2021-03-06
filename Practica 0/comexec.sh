#!/bin/bash
#clear console
clear

#compile the file
mpicc $1.c -o $1

#check if the file exists
if [ -f $1 ]; then
    #run the file
    if [ $# -eq 1 ]; then
        mpiexec -np 8 --oversubscribe ./$1
    else
        mpiexec -np $2 --oversubscribe ./$1
    fi
fi