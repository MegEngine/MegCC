# MegCC 体验
下面将使用 MegEngine 预训练的 shufflenet 模型进行初次使用 MegCC 编译。

## 下载 MegCC 预编译的包

下载提预编译好的 MegCC 模型编译器 [下载 MegCC 编译器](https://github.com/MegEngine/MegCC/releases)，然后进行解压，将获得如下的预编译产物：
```
.
├── bin       : MegCC 的主要编译工具，以及包含一些辅助的工具集。
├── example   : 使用 MegCC 进行推理的示例。
├── runtime   : MegCC 运行时的源代码，进行推理的时候，需要和生成的 Kernel 一同编译。
└── script    : 一些实用的帮助脚本，可以帮助用户快速完成模型编译，以及 Kernel 编译等。
```
## bin文件细节
```
bin
├── mgb-importer  : 辅助工具，主要将解析 MegEngine 模型，然后转化为使用 MLIR 定义的对应的 MGB IR 以及输出。 
├── megcc-opt     : 辅助工具，主要展示 MegCC 定义的 Pass 或者 MLIR 中预定义的 Pass 的具体细节，以及用于 Debug。
├── mgb-runner    : 辅助工具，用于直接使用 MegEngine 运行模型，用于和 MegCC Runtime 的计算结果进行对比，验证正确性。
└── mgb-to-tinynn : 主要的 MegCC 编译工具，将编译 MegEngine 模型，并输出运行这个模型需要的 Kernel，以及对应优化之后的模型。
```

## 获取模型
首先使用 MegEngine dump 一个预训练好的 shufflenet 模型，这个过程中需要安装 [MegEngine](https://www.megengine.org.cn/) ，安装指令可以参考 MegEngine 文档，安装完成之后，并运行如下 code。

``` python
import numpy as np
import megengine.functional as F
import megengine.module as M
import megengine as mge
import megengine.traced_module as tm
from megengine import jit, tensor

shufflenet = mge.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)

data = mge.Tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

traced_shufflenet  = tm.trace_module(shufflenet , data)
traced_shufflenet .eval()

@jit.trace(symbolic=True, capture_as_const=True)
def fun(data, *, net):
   pred = net(data)
   pred_normalized = F.softmax(pred)
   return pred_normalized

fun(data, net=traced_shufflenet)
fun.dump("shufflenetv2.mge", arg_names=["data"])
```

运行之后将在当前目录下生成 shufflenetv2.mge 模型文件。

## 编译模型
编译模型使用上面的预编译包中 bin 目录下面的 mgb-to-tinynn 工具
* 创建一个用于存放生成的模型和 Kernel 的文件夹
  
  ```
    mkdir shufflenetv2_generate
  ```

* 编译模型
  
  ```
    /path/to/mgb-to-tinynn path/to/shufflenetv2.mge path/to/shufflenetv2_generate --input-shapes="data=(1,3,224,224)"  --arm64
  
  ```
  编译模型时候需要指定需要编译的模型，生成目录以及模型的输入 shape 等，最重要的需要指明目标平台，这里是 arm64，目标是运行在手机中。
  输出一堆 log 之后，将完成编译，编译完成之后会在 shufflenetv2_generate 中生成新的模型，结尾为 shufflenetv2.tiny，另外会生成一堆 Kernels。

## 编译runtime + kernel为可执行文件
上面生成的 kernel 的目标平台是 Arm，因此可以运行在 Linux arm 中，也可以运行在 android Arm 手机中，这里以 android Arm 手机作为展示。
编译 android 平台需要下载 NDK，并设置环境变量 NDK_ROOT。如：

```
    export NDK_ROOT=path/to/android-ndk
```

目前在下载的预编译包中已经包含了 runtime 编译的脚本，runtime/script/runtime_build.py，运行：

```
   python3 runtime/scripts/runtime_build.py --cross_build --kernel_dir path/to/shufflenetv2_generate
```

编译完成之后会在打印：

```
-- Installing: xxx/shufflenetv2_generate/runtime/install/include
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/lite-c
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/lite-c/tensor_c.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/lite-c/network_c.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/lite-c/macro.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/lite-c/global_c.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/lite-c/common_enum_c.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/tinycv_c.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/include/tinynn_callback.h
-- Installing: xxx/shufflenetv2_generate/runtime/install/lib/libTinyNN.a
-- Installing: xxx/shufflenetv2_generate/runtime/install/bin/tinynn_test_lite
```

可以看到头文件，静态库，以及可以运行这个模型的可执行文件 tinynn_test_lite 都已经编译安装好了。如果需要进行进一步集成可以 include 头文件并链接 libTinyNN.a，这里我们将 tinynn_test_lite 拷贝到手机上，并把 path/to/shufflenetv2_generate/shufflenetv2.tiny 都拷贝到手机上同一目录下面，运行

```
    tinynn_test_lite shufflenetv2.tiny
```

可以看到推理输出。

## 如果用于编译的模型是 ONNX 模型

如果是输入模型是 ONNX 模型，则需要先将 ONNX 模型转换为 mge 模型，可以使用 [mgeconvert](https://github.com/MegEngine/mgeconvert#13-sparkles-onnx%E6%A8%A1%E5%9E%8B%E4%BA%92%E8%BD%AC) 进行模型转换。具体使用可以参考 mgeconvert。



