## runtime

### build

MegCC runtime目前只支持Cmake编译，目前也支持android编译。编译之前需要将配置好NDK环境。
目前runtime的模型使用flatbuffer序列化的模型，模型格式定义在schema/model.fbs中，runtime是
纯C的工程，所以使用flatcc来载入模型。目前没有支持Cmake中自动编译flatcc，并生成对应的header
的功能。
- 下载flatcc, 在MegCC的根目录运行：git submodule update --init ../third_party/flatcc
- 环境变量中添加环境变量：NDK_ROOT。如：export NDK_ROOT=path/to/NDK/android-ndk-r21d
- 进入MegCC/rumtime,
  需要准备编译的kernel路径，通过mgb-to-tinynn把模型dump到models/mobilenetv1/kernels
- 执行 python3 ./scripts/runtime_build.py --cross_build --kernel_dir ../models/mobilenetv1/kernels  将完成编译，默认编译为静态库，Release版本，非asan检查

目前编译脚本支持如下编译选项
- 编译动态库：python3 ./scripts/runtime_build.py --build_shared_library --cross_build --kernel_dir ../models/mobilenetv1/kernels --remove_old_build
- 静态链接所有libc：python3 ./scripts/runtime_build.py --build_achieve_all --cross_build --kernel_dir ../models/mobilenetv1/kernels --remove_old_build
- 开启asan检查：python3 ./scripts/runtime_build.py --build_with_asan --cross_build --kernel_dir ../models/mobilenetv1/kernels --remove_old_build
- 编译debug版本：python3 ./scripts/runtime_build.py --build_for_debug --cross_build --kernel_dir ../models/mobilenetv1/kernels --remove_old_build

### 运行POC

- 将上述编译好的tinynn_test_lite拷贝到目标机器上
- 将MegCC/model/mobilenet/mobilenet.tiny和输入数据MegCC/runtime/test/resource/input.bin都拷贝到目标手机上
- 执行 ./tinynn_test_lite ./mobilenet.tiny output_dir 0 input.bin 运行，如果需要看最后一层输出的结果，则执行：./tinynn_test_lite ./mobilenet.tiny output_dir 1 input.bin

## ycm 支持
如果你使用vim，你可以使用YCM来进行代码补全和跳转，你只需要执行：
- cd runtime
- building the runtime
