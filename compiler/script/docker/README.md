## preparation

1. we currently provide pre-installed and mlir in docker image to avoid
building them on each pipeline for shorten testing time

    - host-side prebuilt libraries for building and checking compiler itself

      run `./build_mlir.sh` for building and installing mlir/megbrain

## build image

    $ docker build \
        -t registry.hh-d.brainpp.cn/megvii-engine/megbrain_ci:megcc_test_runner_${version} \
        -f /path/to/megcc/compiler/script/docker/Dockerfile \
        /path/to/workspace
