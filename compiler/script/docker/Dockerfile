FROM ubuntu:20.04

RUN apt-get update
RUN TZ=Etc/UTC apt-get -y install tzdata
# install env
RUN apt-get install -y clang llvm-dev
RUN apt-get install -y sudo \
    gcovr autoconf autotools-dev automake build-essential pkg-config apt-utils wget \
    curl zip unzip zlib1g-dev lib32z-dev gnupg flex bison gperf git-core git-lfs swig \
    vim libc6-dev-i386 lib32ncurses5-dev libgl1-mesa-dev libxml2-utils xsltproc libtool \
    librdmacm-dev rdmacm-utils libmpc-dev g++ g++-multilib gcc gcc-multilib python3-dev \
    python3-pip python3-numpy python3-setuptools openssh-server openssh-client rsync \
    software-properties-common iputils-ping iproute2 net-tools inetutils-traceroute \
    libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
    xz-utils libffi-dev liblzma-dev python-openssl libsm6 libxext6 ninja-build cmake qemu \
    libncurses5 qemu-system-arm \
    && apt-get clean \
    && rm -rf /var/lib/apt-get/lists/*

# copy gcc-arm-none-eabi-10.3-2021.07.tar.gz
RUN cd /root && wget http://brain-ftp.megvii-inc.com/gcc-arm-none-eabi-10.3-2021.07.tar.gz && tar -xvf gcc-arm-none-eabi-10.3-2021.07.tar.gz && rm gcc-arm-none-eabi-10.3-2021.07.tar.gz

# copy gcc-arm-10.3-2021.07-x86_64-aarch64-none-elf.tar.gz
RUN cd /root && wget http://brain-ftp.megvii-inc.com/gcc-arm-10.3-2021.07-x86_64-aarch64-none-elf.tar.gz && tar -xvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-elf.tar.gz && rm gcc-arm-10.3-2021.07-x86_64-aarch64-none-elf.tar.gz

# copy NDK
RUN cd /root && wget http://brain-ftp.megvii-inc.com/android-ndk-r21-linux-x86_64.zip && unzip -X android-ndk-r21-linux-x86_64.zip && rm android-ndk-r21-linux-x86_64.zip

# copy freeRTOS repo
RUN cd /root && wget http://brain-ftp.megvii-inc.com/FreeRTOS_19b2ed4bf617018b6.tar.gz && tar -xvf FreeRTOS_19b2ed4bf617018b6.tar.gz && rm FreeRTOS_19b2ed4bf617018b6.tar.gz

# copy optee repo
RUN cd /root && wget http://brain-ftp.megvii-inc.com/optee_repo.tar.gz && tar -xvf optee_repo.tar.gz && rm optee_repo.tar.gz

# copy riscv compiler
RUN cd /root && wget http://brain-ftp.megvii-inc.com/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz && tar -xvf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz && rm Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.6-20220516.tar.gz

# more packages to build qemu and linux kernel for optee
RUN apt-get -y install uuid-dev acpica-tools libglib2.0-dev python3-pyelftools bc python libpixman-1-0 cpio libpixman-1-dev xterm
RUN python3 -m pip install cryptography
# init optee build to generate build cache, in CI env, will use incremental compilation
RUN cd /root && cd optee/build && make -f qemu_v8.mk all -j$(nproc)

# config git config for auto clone git-lfs when init submodule
RUN git lfs install

# install gitlab-runner
RUN curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | bash
RUN apt-get install gitlab-runner=13.12.0 -y

# install llvm-lit for testing
RUN pip3 install lit

# copy prebuilt megengine & mlir
COPY llvm-project /root/llvm-project

# allow normal user to access root privilege
RUN echo "engine-bot  ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
