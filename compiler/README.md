## building

megcc depends on llvm-project and megbrain, now llvm-project and megbrain are integrated by submodule, and static linked into megcc
- update third-party submodule
    ```bash
    $ cd megcc
    $ ./third_party/prepare.sh
    ```
- build megcc
    ```bash
    $ cd megcc/compiler
    $ mkdir build
    $ cd build
    $ cmake .. -G Ninja
    $ ninja
    ```

after build, the compiler tools are stored in build/tools/...


### kernel test
Ninja is recommended(make is not tested ...)
X86 test
```bash
$ mkdir -p build_x86
$ cd build_x86
$ cmake ../test/kernel  -G Ninja
$ ninja
$ ./megcc_test_run
```
ARM Android test 
```bash
$ mkdir -p build_arm
$ cd build_arm
$ cmake ../test/kernel -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake"  -DANDROID_NDK="$NDK_ROOT" -DANDROID_ABI=arm64-v8a  -DANDROID_NATIVE_API_LEVEL=21  -G Ninja -DCMAKE_BUILD_TYPE=Debug
$ ninja
$ copy2phone megcc_test_run
$ run_phone ./megcc_test_run
```
set environment `export extra_gtest_args="--gtest_filter=AARCH64.ConvBiasNCHWNCHW44"` will build specific kernel to save time. 

### regression test
`ninja megcc-test` or `make megcc-test`

ensure `lit` was already installed:

```bash
$ pip3 install --user lit
```

and add option `-DLLVM_EXTERNAL_LIT=/path/to/lit` when cmake build

## ycm support
if you are a vimer, you can use ycm, you just need
- cd compiler
- mkdir build
- building the whole compiler
