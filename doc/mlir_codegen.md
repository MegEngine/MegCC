# 当前mlir codegen能力
KernelGen目录中AUTO_BAREMETAL来标识自动生成的算子, 提供mlir接口和megcc接口的转换, 并调用CodeGen提供的kernel生成接口, 生成的.o最终通过dep传出.  
CodeGen提供生成的kernel的cname和编译好的.o文件(以vector u8提供).  
CodeGen的生成过程分为3步:  
1、构造mlir的计算表示. 创建为一个module对应.o, 创建一个func为实际的计算函数, 在func中填入以memref表示的计算逻辑(linalg表示, 不用tensor是因为转换为memref的pass会把output融在input里). 能用arith尽量不要用math(math的某些ir只能调用外部库来实现, 无法做到.o无依赖)  
2、对计算表示做优化, 并lower到llvm ir. 优化这步主要包含分块和向量化(现在还没做), lower步骤为 tosa->linalg->loop->std->llvm, 注意arith要expand才能被正常lower到llvm, pass中使用了emit-c来把func参数中拆开的memref给缝合起来  
3、编译llvm ir为.o文件. 步骤为创建target和ExcutionEngine(llvm编译和jit的一层mlir封装, 脱离megcc的环境可以通过mlir-translate --mlir-to-llvmir创建bytecode再用clang编译bytecode)  
