#!/usr/bin/bash -e

cd $(dirname $0)
docker build -t megcc_runtime_build_ubuntu -f Dockerfile .
