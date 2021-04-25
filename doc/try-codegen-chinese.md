# 踩坑深度学习kernel自动生成
深度学习kernel是深度学习算子在具体平台上的实现。 运行深度学习模型需要巨大的算力支持, 因此深度学习kernel要尽量挖掘计算平台的性能潜力。 写高性能的kernel并不容易, 一根根献祭头发才写出性能不错的高级语言版本。 精调性能又要细致考察硬件的各种指标， 最终将高级语言改为汇编。 后续的开发者会把这段汇编好好封装并供奉起来, 绝不愿意逐行理清里面的各种技巧和取舍(除非有bug)。   
深度学习有个提高算力利用率的法宝叫算子融合, 将相邻的算子合并为一个（如conv+relu=conv_relu）。算子融合不会减少计算量，但可以减少内存吞吐（中间结果在寄存器中或cache里）。一般硬件系统内存吞吐带宽都远小于计算吞吐带宽，融合后系统短板往往会从访存转移到计算上，有利于打满计算吞吐。极致地融合会导致算子数量爆炸，假设有n个算子，做到任意两个算子融合就需要写n*n种实现；任意三个算子融合则是n的三次方。因此工程上往往只融合一些常见的组合，如conv+relu, elemwise mul + elemwise add。 
如果能机器自动生成高性能深度学习kernel, 发际线就不会变成地平线了。模型编译时机器自动融合算子并生成对应的kernel也能更极致地压榨硬件性能。  

<div  align="center">    
<img src="https://i0.hdslb.com/bfs/face/49cec3657ebb3358519884d27f1f8f7ed4fb880a.jpg@240w_240h_1c_1s.webp" width = 20% height = 20%/>
</div> 

## 深度学习kernel编译器
kernel自动生成的一种方案是深度学习kernel编译器。 类比clang编译器, 工程师用某种中间表示(IR)描述卷积的计算过程, 将其编译为各厂商提供的设备底层表示(如汇编), 这样同一个卷积描述就可以运行在各种平台上。 主流的kernel生成框架都是这样的模式, TVM通过tir描述运算, MLIR通过linalg/scf描述运算, taichi通过限定的类python IR描述运算。 这些编译器能在IR层面融合算子，生成对应的融合kernel。  

进一步，要生成高性能kernel, 不仅是一个能跑的kernel。 如果硬件原生IR足够高级（如NPU提供的硬件卷积IR）, 把网络中卷积的IR直接翻译为硬件的卷积IR, 卷积的执行就是高性能的(如果性能还不够高就是硬件问题了...)。   
实际情况是越通用的硬件(通用意味着研发成本可以被不同场景分摊)提供的IR越低级, 如CPU只提供标量/向量级别的四则运算。 这时编译器把卷积运算的各类迭代转换为for循环, pad操作转换为分支判断, stride转换为迭代参数等等。 一顿转换猛如虎，性能只有百分之五。  

<div  align="center">    
<img src="https://p1.itc.cn/images01/20210517/888f32f98f594099b2228c15eca072d6.gif" width = 80% height = 80%/>
</div>   

细碎平常的卷积实现并不能驱动硬件高效运行。 要让硬件高效执行卷积操作还需要考虑硬件的各级缓存下计算和IO的并行, 对问题进行分块、向量化、对输入进行pack等操作。 这些操作的信息并不来自卷积定义本身, 因此**单凭卷积定义一定无法生成高性能卷积kernel**。  
高性能kernel的生成必须要提供硬件相关信息。 主要的提供方式有三种，编译器硬编码各种pass（[IREE](https://github.com/iree-org/iree)等基于MLIR的编译器常见，类似于工程师的手工优化模板）、基于实际硬件测速的参数搜索（autotvm等）、基于cost模型的参数搜索（[roller](https://www.usenix.org/system/files/osdi22-zhu.pdf)等）。  

硬编码pass可以理解为把生成kernel的各种关键参数（分块大小）的计算方式人工写好。优点是生成速度快，缺点是不同设备需要编译器开发者人工适配，且性能调优需要有编译器开发经验。  
基于硬件测试的参数搜索是通过在实际设备上测试不同参数生成kernel的性能，选出性能较优的参数。 优点是极致性能，不用在编译器层面人工适配不同的设备，缺点是慢且强设备依赖。  
基于cost模型的参数搜索则是在硬件抽象层（cost模型）上预测不同参数的性能，cost模型需要在目标设备上生成。优点是不用在编译器层面人工适配不同设备，且有较快的生成速度。缺点是性能不太靠谱，不同设备cost模型的生成需要手工完成（有个模型市场或许能解决这个问题）。 

总结一下生成深度学习kernel的方案。生成高性能的深度学习kernel需要四类信息，kernel的计算逻辑，负载信息（如矩阵乘的具体shape），优化逻辑，硬件信息。生成思路是计算逻辑提供kernel的base版本，优化逻辑在负载信息的提示下对base版本做等价变换，其中优化逻辑的优化参数还需要适配硬件信息，最终得到一个优化后的kernel。  大部分编译器IR都能完整地表达kernel的计算逻辑。优化逻辑比较复杂，基于控制流的优化有循环拆分、循环合并、循环展开、交换循环顺序等；基于数据流的优化有多级分块、向量化、动态规划等。工程上这些优化手段覆盖了大部分场景，如果能让机器尝试合法地任意组合这些优化逻辑很有希望能找到比手工优化更好的方案。硬件信息可以通过实际测试kernel性能，硬编码、厂商提供标准模型、构造任意的cost模型来提供。目前深度学习kernel的编译器都会提供上述四件套的一些组合和对应实现，不同的编译器提供的组合不同且实现的完成度不同。  

## 踩坑记录 
记录集成算子生成框架的踩坑过程。  
补充一些MLIR中一些基础概念和常用IR的说明, 详情见[官方文档](https://mlir.llvm.org/docs/Dialects/)：  
* MLIR的逻辑是可以通过不同层次的IR（抽象方式）来描述同一个计算过程（深度学习网络的计算图）。生成算子的过程是输入算子计算逻辑高级抽象，逐步把高级抽象转换为贴近硬件的底层抽象，最终生成能直接作为编译器输入的抽象，编译得到对应的算子。优化和转换的手段叫做pass，可以理解为一种处计算图替换操作。优化(transform)是在一种抽象内部根据一些先验知识做的等价变换，如在标准控制流的抽象下做循环顺序的交换。交换(convert)是在不同抽象下的等价替换，切换计算图的表达方式，以期望得到新的优化空间或更接近编译器要求的输入的表达方式。如把标准控制流的抽象转换为llvm的抽象，更接近llc要求的输入。   
* `memref`是内存buffer的抽象，包含buffer的shape和stride信息，与`Tensor`的区别是不用满足单赋值要求。  
* `linalg`是线性代数的抽象，如`linalg.matmul`，具体的计算用迭代器写法的`linalg.generic`表示     
* `scf`是标准控制流的抽象，如`scf.for`, `scf.if`, 相较于迭代器的写法`scf`确定了迭代的顺序   
* `llvm`是通用机器语言的抽象，如`llvm.call`, `br`（跳转）, 相较于标准控制llvm的表达方式更接近于硬件，for循环变成了条件跳转   
* `gpu`是SIMT编程模式的抽象，如`gpu.thread_id x`，相较于`scf`是编程模式的转变，linalg中的并行迭代索引能映射到gpu线程id    
* `spirv`简记`spv`是跨平台SIMT的抽象，可以使用vulkan执行`spirv`的kernel。由于其kernel需要使用全局的资源当参数，有代表性的表示是`spv.GlobalVariabl @xxx bind(0,0)`  
* `vulkan`是一个能运行`spirv` kernel的执行器，MLIR中并没有`vulkan`ir，但`spv`的默认执行方式是`vulkanLaunch`  

MLIR提供代码生成能力，但MLIR没提供生成高性能的矩阵乘/卷积等算子的完整方案。常规的做法是构造一个基于`memref`的`linalg.MatmulOp`对象，然后执行`linalg->scf->llvm`转换流程，配合`ArithmeticExpand`(展开数学计算的细节，如max转换为compare和select)+`FinalizingBufferize`(tensor转换为memref并确认没有tensor只有memref)+`MemRefToLLVM`(把memref转换为llvm类型， llvm.struct)+`ArithmeticToLLVM`(数学运算转换为llvm指令)的pass，时不时地插入一些`Canonicalizer`(规范化，如把`x+x^3+1+x^2`变为`x^3+x^2+x+1`，让后续的pass处理起来容易一些)+`CSE`(相同表达式消除，减少重复运算)来规范化中间IR，最后补一个`ReconcileUnrealizedCastsPass`(忽略未实现的转换语句，由于pass的可以自由调用，可能会出现某些转换语句未实现)。为了能通过C接口调用生成的kernel需要给`LowerToLLVMPass`(把接口和其他ir转换为llvm)打开`emitCWrappers`的开关，否则生成的kernel接口过于琐碎。优化kernel的方法是加优化pass，如LinalgTilingPass和vector相关的pass等。目前的优化问题是LinalgTilingPass无法用于动态shape，没有好用的循环交换和向量化pass，pack也不好做。用现在MLIR的pass优化一个矩阵乘不比手工优化容易。想象中MLIR把各种常规的优化pass都做的比较完善了，两个复杂的算子（如两个矩阵乘）仍然需要重写pass，只有两个矩阵乘的定义可以复用。貌似并不划算。  

基于MLIR的IREE提供了[现成的优化pass](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.cpp#L412)，但集成意义不大。 CAPI层面集成IREE和集成一个预测引擎区别不大，无法做到想要的算子融合和kernel生成。MLIR层面集成IREE能复用手动优化的pass，但和我们自己手写区别不大且融合也不好做。  

探了下MLIR下spirv的kernel生成和vulkan运行，勉强能用但性能存疑。主要流程是`linalg->gpu->spirv->vulkan`。以下用mlir的文本演示两个tensor相加操作的转换过程
```mlir
# 原始linalg IR形式，用generic描述迭代顺序，迭代类型为并行
  func.func @my_codegen_elem(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) {
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0 , %arg1 : memref<8xf32>, memref<8xf32>) outs(%arg2 : memref<8xf32>) {
    ^bb0(%xarg0: f32, %xarg1: f32, %xarg2: f32):
      %0 = arith.addf %xarg0, %xarg1 : f32
      linalg.yield %0 : f32
    }
    return
  }
# 转换为gpu IR形式，并行迭代被映射到block_id上
  gpu.func @my_codegen_elem_kernel(%arg0: index, %arg1: index, %arg2: memref<8xf32>, %arg3: memref<8xf32>, %arg4: memref<8xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<1> : vector<3xi32>>} {
      %0 = gpu.block_id  x
      %1 = arith.muli %0, %arg0 : index
      %2 = arith.addi %1, %arg1 : index
      %3 = memref.load %arg2[%2] : memref<8xf32>
      %4 = memref.load %arg3[%2] : memref<8xf32>
      %5 = arith.addf %3, %4 : f32
      memref.store %5, %arg4[%2] : memref<8xf32>
      gpu.return
    }
# spirv IR形式，和gpu的区别主要是编程风格
  spv.func @my_codegen_elem_kernel(%arg0: !spv.ptr<!spv.struct<(!spv.array<8 x f32, stride=4> [0])>, ...) "None" attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<1> : vector<3xi32>>, workgroup_attributions = 0 : i64} {
      %0 = spv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi32>
      ...
      %5 = spv.Load "StorageBuffer" %4 : f32
      %cst0_i32_1 = spv.Constant 0 : i32
      %cst0_i32_2 = spv.Constant 0 : i32
      %cst1_i32_3 = spv.Constant 1 : i32
      %6 = spv.IMul %cst1_i32_3, %1 : i32
      %7 = spv.IAdd %cst0_i32_2, %6 : i32
      %8 = spv.AccessChain %arg1[%cst0_i32_1, %7] : !spv.ptr<!spv.struct<(!spv.array<8 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      %9 = spv.Load "StorageBuffer" %8 : f32
      %10 = spv.FAdd %5, %9 : f32
      ...
      %13 = spv.AccessChain %arg2[%cst0_i32_4, %12] : !spv.ptr<!spv.struct<(!spv.array<8 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      spv.Store "StorageBuffer" %13, %10 : f32
      spv.Return
    }
# vulkan调用形式， spirv_blob为二进制形式的spirv代码
  llvm.func @my_codegen_elem_kernel(...) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.mlir.constant(8 : index) : i64
    llvm.call @vulkanLaunch(...) {spirv_blob = "...", spirv_entry_point = "kernel_add"} : (...) -> ()
    llvm.return
  }
```
需要注意`linalg`直接到`spirv`的转换pass目前不好用，需要借助`gpu`IR完成`scf`的parallel到thread id的映射。脱离MLIR执行`vulkan`+`spirv`的kernel还需要用`vulkan`的CAPI完成三个MLIR默认接口`initVulkan`, `setEntryPoint`, `runOnVulkan`的封装。   
测了一把ncnn的vulkan实现和其他框架的opencl实现，发现vulkan测出来性能都不大行，不知道有没有大佬分析一波。如果手动优化的vulkan跑不过opencl，用MLIR生成spirv的kernel替换手工优化的opencl kernel可能会得不偿失。   

集成TVM，由于离线编译时无法要求用户提供运行环境，只能考虑预先生成kernel。这要求kernel的泛用性，不能每种shape都重新生成一个kernel。TVM中可以通过var的抽象生成支持shape变化的kernel，但[auto schedule目前不支持var](https://discuss.tvm.apache.org/t/can-tvm-support-auto-scheduler-with-te-var/10917/2)和[全局动态shape](https://discuss.tvm.apache.org/t/does-tvm-support-dynamic-input-shape/11069/6)。或许过段时间，大佬们给TVM补上auto+var后，这种集成方式才真的可行。  

## 结语  
如上，离线编译环境下自动生成高性能kernel的尝试并不成功。对于白嫖党来讲，MLIR相关pass并不成熟，且不具备自动获取硬件信息的能力。TVM对动态shape的支持还有待加强。等待各框架的持续完善中...（为啥白嫖，支持业务把时间都花在手工优化上了，在收益有限的预期下没法投太多时间把这些框架的不足给补上）  

<div  align="center">    
<img src="https://github.com/qq332982511/onnx_model/raw/master/baipiao.jpg" width = 40% height = 40%/>
</div> 

## 一些脑洞  
程序的表达能力上限是汇编，汇编指令分为控制流指令和数据流指令。如果优化逻辑覆盖这两类的所有指令，让机器合法任意地组合优化逻辑一定可以逼近程序能达到的性能极限。再要突破就只能改硬件了...  
如果矩阵能表达任意语义，让计算逻辑（base版kernel）为矩阵A，优化逻辑为矩阵B，适配负载信息和硬件信息的优化逻辑参数为矩阵C。D=B@A+C，根据线性变换含义D就是优化后kernel矩阵。不知道用这种建模方式能不能构造一套形式kernel生成模型...
