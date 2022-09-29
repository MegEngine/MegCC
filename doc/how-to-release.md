# How to Release 
MegCC depends some third party projects, include:
- MegEngine
- llvm-project
- flatbuffer
- flatcc
- googletest

When release all the third party will build into MegCC, for convenience，MegCC provides a script [release_megcc.sh](../script/docker_release_megcc.sh) to build and pack the release file.

## Requirement
- Cmake above 3.15.2
- Ninja
- gcc

## Release

1. Modify major/minor/patch in compiler/include/compiler/Common/Version.h.in
2. Write changelog https://discourse.brainpp.cn/t/topic/56470
3. Git tag branch
4. Run the script [release_megcc.sh](../script/docker_release_megcc.sh) in [docker image](../script/docker/Dockerfile) with a directory in which all release file will be generated. Finally a tar file named `megcc_relase_*.tar.gz` will created by compressing all file in the release directory.

```bash
cd /path/to/megcc
# build docker image, if built skip it
docker build script/docker -t megcc_manlylinux2014
# run release script
./script/docker_release_megcc.sh
```
5. copy megcc release target to OSS
```bash
# copy release target to megcc 
oss cp megcc_release_xxx.tar.gz s3://megengine-built/megcc/megcc_release_xxx.tar.gz 
```
6. Send email to notify the users of megcc