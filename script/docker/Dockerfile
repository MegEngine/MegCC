FROM quay.io/pypa/manylinux2014_x86_64:2020-12-31-56195b3

ARG platform
COPY setup_mirror.sh .
RUN ./setup_mirror.sh "$platform"
RUN yum install -y pcre-devel devtoolset-9 vim-common git-lfs ninja-build python3 devtoolset-9-libatomic-devel
RUN git lfs install

RUN git config --global user.name "xxxxxname"
RUN git config --global user.email "xxxx@example.com"
