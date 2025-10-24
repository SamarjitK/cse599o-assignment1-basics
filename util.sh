#!/bin/bash


### Set initial time of file
CWD=$(pwd)
FPATH="$CWD/slurm/samarjit-slurm-$1.out"
LTIME=$(stat -c %Z "$FPATH")

function print_status {
    clear;
    echo -e "\n";
    squeue;
    echo -e "\n===================   START  ===================";
    cat $1 | head -n 11
    echo -e "\n\n===================   TAIL   ===================";
    cat $1 | tail -n $2;
}

print_status $FPATH $2;

while true
do
  ATIME=$(stat -c %Z "$FPATH")

  if [[ "$ATIME" != "$LTIME" ]] then
    print_status $FPATH $2;
    LTIME=$ATIME
  fi
  sleep 5
done