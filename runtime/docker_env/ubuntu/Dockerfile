FROM ubuntu:20.04

ENV NDK_ROOT=~/android-ndk-r25c

# update
RUN apt-get update -y

# build env
RUN DEBIAN_FRONTEND=noninteractive \
TZ=Etc/UTC apt-get install -y \
pkg-config build-essential git-lfs \
autoconf gdb git-core gnupg flex bison gperf \
zip curl zlib1g-dev gcc cmake ninja-build \
g++ rsync wget vim llvm clang python3 \
gcc-multilib g++-multilib

# download NDK
RUN cd ~/ && wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip && unzip -X android-ndk-r25c-linux.zip && rm android-ndk-r25c-linux.zip
