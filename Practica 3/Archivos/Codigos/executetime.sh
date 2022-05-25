#!/bin/bash
#clear

#check if the file exists
if [ -f $1 ]; then
	for ((A=1;A<512;A=A+5))
	do
        	for ((P=2;P<512;P=P+5))
        	do
            	./$1 $(( $P )) $(( $A )) >> Tiempos.dat
        	done
	done
fi

#parameters: $1 = file_name
