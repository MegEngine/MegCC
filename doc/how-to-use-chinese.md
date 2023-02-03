# MegCC Release 包的目录结构
## 压缩包的一级目录
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
├── mgb-importer    : 辅助工具，主要将解析 MegEngine 模型，然后转化为使用 MLIR 定义的对应的 MGB IR 以及输出。 
├── megcc-opt       : 辅助工具，主要展示 MegCC 定义的 Pass 或者 MLIR 中预定义的 Pass 的具体细节，以及用于 Debug。
├── mgb-runner      : 辅助工具，用于直接使用 MegEngine 运行模型，用于和 MegCC Runtime 的计算结果进行对比，验证正确性。
├── hako-to-mgb     : 辅助工具，用于将使用 hako 打包之后的模型转换为 MegEngine 对应的模型。
├── mgb-to-tinynn   : 主要的 MegCC 编译工具，将编译 MegEngine 模型，并输出运行这个模型需要的 Kernel，以及对应优化之后的模型。
└── kernel_exporter : 辅助工具，用于指定 kernel C 代码的导出。
```

# 使用 MegCC 完成模型部署
首先需要从 [github](https://github.com/MegEngine/MegCC/releases) 上下载需要的 MegCC 发版包，然后解压这个压缩包：`tar -xvf megcc_release_*.tar.gz`。使用 MegCC 完成模型部署主要步骤有三个步：
- 模型编译：编译 MegEngine 模型，生成运行这个模型对应的 Kernel 以及和这些 Kernel 绑定的模型。
- Runtime编译：这个阶段会将 Runtime 和上一步中生成的 Kernel 一起编译成一个静态库。
- 集成到应用中：调用上一步编译的静态库的接口进行推理。

## 模型编译
模型编译阶段主要使用 mgb-to-tinynn 工具，编译完成之后，会在用户给定的目录下面，生成对应的纯 C 代码的 Kernel 以及对应的模型。为了编译模型，mgb-to-tinynn 工具需要用户提供一个 [Json](https://en.wikipedia.org/wiki/JSON) 文件来描述编译的具体细节。

> 目前 MegCC 只支持 mge 模型作为输入，其他模型格式可以考虑转换到 ONNX，然后通过 [mgeconvert](https://github.com/MegEngine/mgeconvert#13-sparkles-onnx%E6%A8%A1%E5%9E%8B%E4%BA%92%E8%BD%AC) 进行模型格式转换。

### 编写 Json 文件
先看一下具体示例吧：
```json
{
    "dump_dir":"./batch_dump/",
    "models":[
        {
            "model_name":"det_nchw44",
            "model_path":"path/to/model.mdl",
            "input_shape_str":"data=(1,1,384,288):data=(1,1,288,384)",
            "enable_nchw44":true
        },
        {
            "model_name":"pf_nchw44",
            "model_path":"path/to/another_model.mdl",
            "input_shape_str":"data=(1,1,112,112)",
            "enable_nchw44":true
        }
    ],
    "cv":{
        "transpose":["ui8"],
        "roicopy":["ui8"],
        "rotate":["ui8"],
        "flip":["ui8"],
        "resize_linear":["ui8"],
        "warp_affine_replicate_linear":["ui8"],
        "rgb2bgr":["ui8"],
        "yuv2bgr_nv21":["ui8"],
        "rgb2yuv":["ui8"]
    }
}
```
- 设置模型编译之后 dump 的路径，可以在 mgb-to-tinynn 工具中通过 --dump 参数进行 override。
- Json 文件中需要指定使用 mgb-to-tinynn 编译的模型名称，模型的路径，以及模型的输入数据，以及一些优化参数等
  - 如果部署的实际情况中需要对个模型组成 pipline，需要指定多个模型
  - 如果一个模型在实际推理过程中可能需要多种输入 shape，需要分别在 `input_shape_str` 中指定，并用 `:` 分割开。
  - 支持 `enable_nchw44` 和 `enable_nchw44_dot` 两个优化选项，`enable_nchw44` 为 true 表示，优化模型推理中 Tensor layout 为 [NC4HW4](https://cloud.tencent.com/developer/article/1748441)。`enable_nchw44_dot` 为 true 表示，优化模型推理中 Tensor layout 为 [NC4HW4](https://cloud.tencent.com/developer/article/1748441)，并且在推理过程中使用 [ArmV8.2 dot](https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/exploring-the-arm-dot-product-instructions) 指令进行推理加速
- 另外为了方便用户集成时候使用 cv 算子进行模型的前后处理，可以在这个 Json 文件中指定需要用到的 cv 算子的名称以及对应的数据类型。MegCC 支持的 cv 算子 [列表](opr.md)。

### 模型编译
编译模型目前可以使用 mgb-to-tinynn 这个可执行文件完成编译，也可以使用 Release 包里面的现成脚本 `./script/ppl_gen.sh` 进行编译。

#### 使用现成脚本进行模型编译（推荐）
Release 包中的 script 目录下面有一个 `ppl_gen.sh` 的文件，直接执行：
```bash
./script/ppl_gen.sh ./bin/mgb-to-tinynn ./example/mobilenet.json mobilenet_gen --arm64
``` 
`./script/ppl_gen.sh` 这个脚本将执行模型编译，并把 Runtime 需要的资源一同打包在一个压缩包中，方便后续 Runtime 的编译，解压这个压缩包将得到：
```
.
├── example : 在各种操作系统上集成的example
│   ├── Nonstandard_OS
│   │   ├── bare_board
│   │   ├── freeRTOS
│   │   └── tee
│   └── standard_OS
├── flatcc：编译 runtime 时候依赖的 flatcc
├── include ：runtime 的头文件
│   └── lite-c
├── kern ：编译生成的 Kernel 包括 cv 算子
├── model：编译之后生成的模型，用于部署
├── model_info
├── schema
├── script：各种帮助脚本
└── src：runtime 的源文件
    ├── cheader
    ├── lite
    └── vm
```

#### 使用可执行文件编译
使用 mgb-to-tinynn 和上面写好的 Json 文件执行：

```bash
mgb-to-tinynn --json=/path/to/json --[target]
``` 
完成模型编译后，将生成的运行这个模型的 Kernel，和这些 Kernel 绑定的模型文件以及 cv 算子都放在 Json 文件中指定的目录。其中
- target：可以是 baremetal, arm64, armv7, arm64v7.
  
> Note: 
 - baremetal: 生成的 Kernel 为单片机可以运行的纯 C 形式
 - arm64v7: 生成能够同时在 Arm64 和 ArmV7 上可以运行的两套 Kernel 以及他们对应的模型，这时候，模型文件可能会比 target 为 arm64 时候大一点。
  
如编译 Release 包中的 mobilenet 模型，目标机器是 arm64 机器，运行如下命令：
```bash
mkdir mobilenet_gen
./bin/mgb-to-tinynn --json=./example/mobilenet.json --arm64 --dump mobilenet_gen
``` 
将会在 mobilenet_gen 中生成运行这个模型的所有 kernel，以及模型的 input 信息：`mobilenet_nchw44.tiny.txt` 文件中，以及新生成的模型文件：`mobilenet_nchw44.tiny`。

## Runtime 编译
> warning: **编译 Runtime 需要编译机器上有对应的编译环境，比如：编译 Android 上可运行的库，需要环境中有 NDK 工具链，并且设置 NDK_ROOT环境变量为 NDK 的路径。**

针对上面编译模型时候的两种方法，也提供了单独编译的方法和使用脚本编译的方法。

### 使用脚本编译（推荐）
MegCC 提供一个脚本方便完成上述编译操作
> warning: **进行下面的编译，需要环境中有 NDK 工具链，并且设置 NDK_ROOT 环境变量为 NDK 的路径。**
执行：

- 解压上面`使用现成脚本进行编译`之后生成的`megcc_ppl_gen.tar.gz`，执行 `tar -xvf megcc_ppl_gen.tar.gz` 并进入加压之后的目录。
  - 如果编译平台是 arm64，执行 `./ppl_build.sh`。
  - 如果编译平台是 armv7，执行 `./ppl_build.sh -m armeabi-v7a`。

最终运行时需要的模型文件在解压之后的 model 目录下面 `mobilenet_nchw44.tiny`。

### 单独编译
上面使用使用可执行文件编译的 Kernel 和模型保存在指定的生成文件中，为了生成最终的库文件，还需要和 MegCC 预先写好的 Runtime 一同进行编译。
> warning: **进行下面的编译，需要环境中有 NDK 工具链，并且设置 NDK_ROOT 环境变量为 NDK 的路径。**
执行：
```bash
./runtime/scripts/runtime_build.py --kernel_dir ./mobilenet_gen --cross_build
``` 
`runtime_build.py` 脚本默认编译支持 Android 上 aarch64 的编译，用户可以通过：
- `--cross_build_target_arch`: 指定目标编译的平台，目前支持：['x86_64', 'i386', 'aarch64', 'armv7-a', 'cortex-m', 'armv7-a-qemu']。
- `--cross_build_target_os`：指定编译的目标操作系统，目前支持：['ANDROID', 'LINUX', 'IOS', 'NOT_STANDARD_OS']
- `--kernel_dir`：指定编译的 Kernel 路径。

更多参数可以执行 `./runtime/scripts/runtime_build.py --help` 获得。

在目标机器上使用 `tinynn_test_lite` 和编译之后的模型文件 `mobilenet_gen/mobilenet_nchw44.tiny` 以及一定的输入数据就可以进行 mobilenet 模型的推理。

### 编译产物
上面命令执行完成之后将之前生成的 mobilenet_gen 中的 Kernel 和 Release 包中的 Runtime 一同编译并 install 在 mobilenet_gen 的 runtime/install 目录中，目录中文件如下：

```
.
├── bin
│   └── tinynn_test_lite ：集成了下面 libTinyNN.a 的文件，并编译之后的可执行文件，用户可以直接运行
├── include ：集成下面 libTinyNN.a 需要的头文件
│   ├── lite-c
│   │   ├── common_enum_c.h
│   │   ├── global_c.h
│   │   ├── macro.h
│   │   ├── network_c.h
│   │   └── tensor_c.h
│   ├── tinycv_c.h ：使用编译之后的 cv 算子需要的头文件
│   └── tinynn_callback.h
└── lib
    └── libTinyNN.a ：编译 runtime 和编译之后的 Kernel 生成的静态库
```

## 集成
上面编译完成之后生成的产物主要有三个：
- 模型编译之后生成的模型文件，如：`mobilenet_nchw44.tiny`
- 编译之后生成的 libTinyNN.a 库文件。
- 集成时候需要的头文件。

通过上面的产物可以直接在目标程序中进行集成，集成 demo 可以参考 [集成代码](../runtime/example/standard_OS/lite_main.c) 中关于接口的调用方法。

至此整个模型部署流程已经完成。

# Debug 计算错误问题
> warning: **进行下面的步骤，需要环境中有 NDK 工具链，并且设置 NDK_ROOT 环境变量为 NDK 的路径。**

> warning: **下面的步骤是在 android 手机上进行，执行时需要保证 android 设备可以通过 rsync访问** 

进行正确性检查主要使用脚本 [test_model.py](../script/test_model.py) 帮助完成在目标机器上运行 MegCC 编译之后的 Runtime 和直接使用 MegEngine 运行模型的结果进行对比。使用示例：
- 在通过上面 `ppl_gen.sh` 工具生成的 tar 解压之后的目录 `mobilenet_gen` 包中执行 `./ppl_build.sh` 编译运行需要的文件（这里默认编译平台为 arm64）
- 编译完成之后，执行
    ```python3
    python3 script/test_model.py --target user@you_phone  mobilenet_gen --mdl="mobilenet_nchw44:./example/mobilenet.mdl"
    ```  
其中：`--target user@you_phone` 为一个可以通过 [rsync](https://en.wikipedia.org/wiki/Rsync) 访问的 android 设备。运行完成之后的 log 中会报告最终检测的结果，以及精度误差等。 

# 进阶使用
MegCC 是在 MLIR 的基础组件上开发的，你可以通过下面的工具来探索 MegCC 编译过程的更多细节。 
## mgb-to-tinynn
上面已经用到过 MegCC 主要的编译工具 `mgb-to-tinynn` 进行模型编译，下面的命令： 
```bash
./bin/mgb-to-tinynn ./example/mobilenet.mdl --input-shapes="data=(1,3,224,224)" ./dump_kernel --arm64 --enable_nchw44
```
将使用 `mgb-to-tinynn` 编译 `./example/mobilenet.mdl`, 使用的输入 Tensor 的名字是 `data`, shape 是`(1,3,224,224)`， 编译之后将模型和 Kernel 保存在 `dump_kernel` 目录中，编译的目标平台为 `arm64`，打开了 `enable_nchw44` 优化选项，更多细节可以通过运行 `./bin/mgb-to-tinynn --help` 获取。 
- 使用 `--arm64v7` 将编译既可以在 arm64 也可以在 armv7 上运行的 Kernel 和对应的模型，但是模型的体积会稍微比仅仅编译 arm64 或者 armv7 大一点。  
- 使用 `--enable_nchw44_dot` 开启编译 armv8.2 dot 指令优化的 Kernel.    
- 使用 `--save-model` 将把编译之后的模型以数据的形式 dump 在生成的 kernel 中，这个功能在没有文件系统的设备上使用，或者在部署时候不希望处理分离的模型和 runtime 时候使用。

### mgb-importer
`mgb-importer` 主要将解析 MegEngine 模型，然后转化为实用 MLIR 定义的对应的 MGB IR 以及输出，阅读生成的 mlir 的文本形式，你会更加熟悉你的模型。执行： `./bin/mgb-importer example/mobilenet.mdl mobilenet.mlir` 将输出 import 完成之后的 mlir 的文本形式。

### mgb-runner
用于直接使用 MegEngine 运行模型，可以和 MegCC Runtime 的计算结果进行对比，用于验证正确性等。 执行示例：
```bash
 ./bin/mgb-runner ./example/mobilenet.mdl ./mgb_out --input-shapes="data=(1,3,224,224)" --input-data="data=input_1_3_224_224_fp32.bin"
```
其中`./example/mobilenet.mdl` 为原始的 MegEngine 模型，输入 Tensor 的名字为 `data`，数据为 `input_1_3_224_224_fp32.bin`。

### hako-to-mgb
用于将使用 hako 打包之后的模型转换为 MegEngine 对应的模型.  

### megcc-opt
将编译 MegEngine 模型，并输出运行指定 Pass 之后的模型 IR 表示。通过这个工具你可以一步一步的探索 MegCC 的编译细节，以及每一个 Pass 完成之后，mlir IR 发生的变化。MegCC 中使用到的主要 Pass 有：`--MGB-to-Kernel --finalizing-bufferize --memory-forwarding --static-memory-planning` 等。

### kernel_exporter
导出指定 kernel 的 C代码，获取 kernel 对于不同后端的具体实现。使用方法如下：
##### 使用默认 kernel 属性

```bash
./kernel_exporter --arch <arch_type> --kernel <kernel_type> --use_default_attr
```
##### 交互式用户指定 kernel 属性
```bash
./kernel_exporter --arch <arch_type> --kernel <kernel_type>
```

具体 arch_type 和 kenrel_type 可以通过 `--help` 查看。目前支持的 kenrel type有：
```bash
ArgSortKernel           ArgmaxKernel                BatchMatmulKernel       CVTransposeKernel
ConcatKernel            ConvBackDataKernel          ConvKernel              CvtColorKernel
ElemwiseKernel          ElemwiseMultiKernel         FlipKernel              IndexingMultiAxisKernel
IndexingOneHotKernel    MatrixInvKernel             MatrixMulKernel         PoolingKernel
PowCKernel              ReduceKernel                RelayoutKernel          ResizeKernel
RoiCopyKernel           RotateKernel                TopK                    TypeCvtKernel
WarpAffineKernel        WarpPerspectiveKernel
```

