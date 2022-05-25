#!/bin/bash
USER='estudiante15'
PASS='tnapzscrgg'
FILE='output.txt'
PROGRAM='final'
if [ -f $FILE ]; then
    rm $FILE
fi
sshpass -p $PASS ssh $USER@genmagic.ugr.es "rm -rf ~/*"
sshpass -p $PASS scp -r ./Codigos/* $USER@genmagic.ugr.es:~/
sshpass -p $PASS ssh $USER@genmagic.ugr.es "nvcc pr3.cu -o $PROGRAM && ./$PROGRAM 200 200 >> $FILE"
sshpass -p $PASS scp -r $USER@genmagic.ugr.es:~/$FILE ./
cat $FILE
