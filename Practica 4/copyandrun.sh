#!/bin/bash
USER=''
PASS=''
FILE='output.txt'
SOURCE='pr4-1.cu'
PROGRAM='ejer1'
PARAMS=''

if [ -f $FILE ]; then
    rm $FILE
fi
sshpass -p $PASS ssh $USER@genmagic.ugr.es "rm -rf ~/*"
sshpass -p $PASS scp -r ./Codigos/* $USER@genmagic.ugr.es:~/
sshpass -p $PASS ssh $USER@genmagic.ugr.es "nvcc $SOURCE -o $PROGRAM && ./$PROGRAM $PARAMS >> $FILE"
sshpass -p $PASS scp -r $USER@genmagic.ugr.es:~/$FILE ./
cat $FILE
