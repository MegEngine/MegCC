#!/usr/bin/env bash
set -e
cd $(dirname $0)

ISORT_ARG=""
BLACK_ARG=""

while getopts 'd' OPT; do
    case $OPT in
        d)
            ISORT_ARG="--diff --check-only"
            BLACK_ARG="--diff --check"
            ;;
        ?)
            echo "Usage: `basename $0` [-d]"
    esac
done

isort $ISORT_ARG -j $(nproc) . 
black $BLACK_ARG --target-version=py35 . 
isort $ISORT_ARG -j $(nproc) ../tools 
black $BLACK_ARG --target-version=py35 ../tools
