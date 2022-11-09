# how to add a new kerrnel in megcc compiler

## add kernel IR

kernel IR is used to transform your target framework model which can be discribed with MLIR  into megcc model.
The kernel IR is defined in [AbstractKernels.td](../compiler/include/compiler/Dialect/Kernel/IR/AbstractKernels.td) with pattern as follow:

``` mlir
def Pooling2DKernel: AbstractKernelBase<"Pool2d"> {
    let arguments = (ins
        StrAttr:$mode,
        UI32Attr:$pad_h,
        UI32Attr:$pad_w,
        UI32Attr:$stride_h,
        UI32Attr:$stride_w,
        UI32Attr:$window_h,
        UI32Attr:$window_w,
        StrAttr:$format,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}
```

it declare the parameters, inputs and outputs for the kernel used, with this defination the kernel MLIR module will be auto genrated

## add kernel generate template

#### registe kernel type in kernel generate template

the kernel type is registe in [KernelGen.h](../compiler/include/compiler/KernelGen/KernelGen.h), which is as follow:

``` cpp

struct KernelPack {
    enum class KernType {
        Unknow = 0,
        ConvKernel,
        ElemwiseKernel,
        ElemwiseMultiKernel,
        PoolingKernel,
        MatrixMulKernel,
        MatrixInvKernel,
        RelayoutKernel,
        ReduceKernel,
        IndexingMultiAxisKernel,
        IndexingOneHotKernel,
        WarpPerspectiveKernel,
        WarpAffineKernel,
        TypeCvtKernel,
        TopK,
        BatchMatmulKernel,
        PowCKernel,
        CVTransposeKernel,
        FlipKernel,
        ResizeKernel,
        RotateKernel,
        RoiCopyKernel,
        CvtColorKernel,
        ArgSortKernel,
        ArgmaxKernel,
        ConcatKernel,
        InternelKernel,
    };
    static std::pair<std::vector<const KernelFunc*>, const DeduceFunc*>
    GetKernel(KernelPack::KernType kernel_type, Arch arch);
};

```

#### add new kernel generate implementation

the kernel generate implementation is with such pattern:

``` cpp

struct KernelFunc {
    virtual ~KernelFunc(){};
    virtual bool IsAvailable(TContext* context) const = 0;
    virtual KernelPriority GetPriority() const {
        return KernelPriority::NORMAL;
    }
    //! kernel gen
    virtual std::string GetKernelSymbol(TContext* context) const = 0;
    virtual std::string GetKernelSignature(TContext* context) const {
        return GetKernelSymbol(context) + GenCommonCall();
    };
    virtual std::string GetKernelBody(TContext* context) const = 0;
    //! cv gen
    virtual bool IsCVAvailable(TContext* context) const { return false; };
    virtual std::string GetCVKernelSymbol(TContext* context) const {
        return "";
    };
    virtual std::string GetCVKernelSignature(TContext* context) const {
        return "";
    };
    virtual std::string GetCVKernelBody(TContext* context) const { return ""; };

    //! init gen
    virtual std::string GetInitSymbol(TContext* context) const {
        return GetKernelSymbol(context) + "_init";
    };
    virtual std::string GetInitSignature(TContext* context) const {
        return GetInitSymbol(context) + GenCommonInitCall();
    };
    virtual std::string GetInitBody(TContext* context) const {
        std::stringstream ss;
        ss << GenCommonRet() << " " << GetInitSignature(context) << R"({
                if (nr_out_weight){
                    *nr_out_weight = 0;
                }
               return TinyNN_SUCCESS;
            })";
        return ss.str();
    };

    //! workspace gen
    virtual std::string GetWorkspaceSymbol(TContext* context) const {
        return GetKernelSymbol(context) + "_workspace";
    };
    virtual std::string GetWorkspaceSignature(TContext* context) const {
        return GetWorkspaceSymbol(context) + GenCommonWorkspaceCall();
    };
    virtual std::string GetWorkspaceBody(TContext* context) const {
        std::stringstream ss;
        ss << GenCommonRet() << " " << GetWorkspaceSignature(context) << R"({
               return TinyNN_SUCCESS;
            })";
        return ss.str();
    };
    //! if get workspace need Jit execute, it should not depend on extern
    //! function
    virtual std::string GetWorkspaceBodyAndJitExec(TContext* context) const {
        return GetWorkspaceBody(context);
    };
    //! All body will be warp by guard begin, guard end
    virtual std::string GetBodyGuardBegin(TContext* context) const {
        return "";
    }

    virtual std::string GetBodyGuardEnd(TContext* context) const { return ""; }

    //! The internal kernel used by the kernel function
    virtual std::vector<KernelObj> GetDependInternalSymbol(TContext*) const {
        return {};
    }
};
```

you need to inherit this class and implement the pure virtual function:

``` cpp
    // the kernel available condition
    virtual bool IsAvailable(TContext* context) const = 0;
    // the kernel call name without argument
    virtual std::string GetKernelSymbol(TContext* context) const = 0;
    // the kernel implement body which is wirte with string
    virtual std::string GetKernelBody(TContext* context) const = 0;
```

if other functions need to override, you must override it

#### add new kernel in kernel package

the kernel package is defined in [KernelPack.cpp](../compiler/lib/KernelGen/BareMetal/KernelPack.cpp):

``` cpp
##include "Pooling.h"
struct AllBareKernel {
    AllBareKernel() {
        inner_map[KernelPack::KernType::PoolingKernel] = {
                std::make_shared<BareMetal::PoolingKernel>()};
    }

    std::unordered_map<KernelPack::KernType,
                       std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};
```

## add kernel test

#### add testcase for kernel

kernel is test with gtest, add testcase to test the rightness of kernel
the test file is in **`compiler/test/kernel/opr/xxx/<kernel_type>.cpp`**, for example

```cpp
using Mode = PoolingForward::Param::Mode;
TEST(NAIVE, PoolingNCHW) {
    Checker<Pooling> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    PoolingForward::Param param;
    checker.set_param(param);
    for (auto mode :
         {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING})
        for (size_t window : {2, 3, 5})
            for (size_t stride : {(size_t)1, window})
                for (size_t pad : {(size_t)0, window / 2})
                    for (size_t n : {1, 3})
                        for (size_t c : {1, 3})
                            for (size_t hw : {5, 23}) {
                                param.mode = mode;
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.window_h = window;
                                param.window_w = window;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs({{n, c, hw, hw}, {}});
                            }
}
```

#### add test module implementation

after the testcase is added already, you need to add checker, megcc test porxy, attribute convert function, benchmarker and workload proxy implementation for test, which is separtely defined in [checker.cpp](../compiler/test/kernel/common/src/checker.cpp), [cc_proxy.cpp](../compiler/test/kernel/common/src/cc_proxy.cpp), [cc_fill_attr.cpp](../compiler/test/kernel/common/src/cc_fill_attr.cpp), [benchmark.cpp](../compiler/test/kernel/common/src/benchmark.cpp) and [workload_proxy.cpp](../compiler/test/kernel/common/src/workload_proxy.cpp) for example:

```cpp
// template specilization compiler/test/kernel/common/src/checker.cpp
template class Checker<megdnn::PoolingForward>; 

// add megcc test proxy compiler/test/kernel/common/src/cc_proxy.cpp 
##define DEF_CCOPRPROXY(_OPR_CLS)                                               \
    template PerformanceResult CCOprProxy<_OPR_CLS>::exec(                     \
            _OPR_CLS* opr, const TensorNDArray& tensors, KernelGen::Arch arch, \
            const BenchmarkOption& benchmark_option,                           \
            const std::string& kernel_symbol,                                  \
            const std::unordered_map<std::string, CCAttr>& proxy_attr,         \
            bool gen_dynamic)

DEF_CCOPRPROXY(megdnn::PoolingForward);

// attribute converter compiler/test/kernel/common/src/cc_fill_attr.cpp
template <>
KernelGenRet opr_fill_attr<megdnn::ConvolutionBackwardData>(
        std::unordered_map<std::string, CCAttr>& attr_map,
        megdnn::ConvolutionBackwardData* opr, const TensorNDArray& tensors,
        KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr) {
    auto param = opr->param();
    uint32_t kh = 0, kw = 0;
    get_kernel_size(kh, kw, tensors[0], param.sparse, param.format);
    attr_map["kernel_h"] = CCAttr(kh);
    attr_map["kernel_w"] = CCAttr(kw);
    FILL_MAP(attr_map, param, stride_h);
    FILL_MAP(attr_map, param, stride_w);
    FILL_MAP(attr_map, param, pad_h);
    FILL_MAP(attr_map, param, pad_w);
    FILL_MAP(attr_map, param, dilate_h);
    FILL_MAP(attr_map, param, dilate_w);
    FILL_MAP_EX(attr_map, param, sparse, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, format, dnnparam_2_str);
    FILL_MAP_EX(attr_map, param, mode, dnnparam_2_str);
    return KernelGen::KernelPack::GetKernel(KernType::ConvBackDataKernel, arch);
}

// benchmarker compiler/test/kernel/common/src/benchmark.cpp
template class megcc::test::Benchmarker<megdnn::PoolingForward>;

// workload proxy compiler/test/kernel/common/src/workload_proxy.cpp
template <>
size_t WorkloadOprProxy<megdnn::PoolingForward>::get_compute_workload(
        megdnn::PoolingForward* opr, const TensorNDArray& tensors) {
    auto param = opr->param();
    auto dst_layout = tensors[1].layout;
    float computation =
            dst_layout.total_nr_elems() * param.window_h * param.window_w;
    return computation;
}


```

## modify the convert pass

the convert pass is used to convert your target framework IR into megcc IR, this include attribute and kernel convertion

#### add attirbute convert function

for example, when you need to convert megengine model into megcc, and there is a new kernel need added in megcc, the attribute convertion is with pattern as follow:

``` cpp
template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Pooling>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    using Mode = ::megdnn::param::Pooling::Mode;
    using Format = ::megdnn::param::Pooling::Format;

    SmallVector<NamedAttribute, 4> attrs;
    GetParam("stride_h");
    GetParam("stride_w");
    GetParam("pad_h");
    GetParam("pad_w");
    GetParam("window_h");
    GetParam("window_w");

    GetParamEnum(Mode, "mode");
    GetParamEnum(Format, "format");

    return attrs;
}
```

this is added in [MGBToKernelHelper.h](../compiler/lib/Conversion/MGBToKernel/MGBToKernelHelper.h)

#### add kernel convert funtion

the pattern  which is in [MGBToKernel.cpp](../compiler/lib/Conversion/MGBToKernel/MGBToKernel.cpp) is as follow:

``` cpp
template <class SrcOp, class DstOp>
class GenericConverter : public OpConversionPattern<SrcOp> {
public:
    using OpAdaptor = typename SrcOp::Adaptor;
    using OpConversionPattern<SrcOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(
            SrcOp op, OpAdaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "General convert MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        auto attrs =
                ConvertAttr<SrcOp>(op->getAttrDictionary(), op->getContext());
        return createOp<DstOp>(op, operands, rewriter, attrs);
    }
};
```

## register kernel in transform pass

 the kernel is only registed in [KernelRegister.h](../compiler/lib/Dialect/Kernel/Transforms/KernelRegister.h) that can be generated actually. you need add two key objects, one is getkernel object , the other is buildintemplateOpr object, for example:

```cpp

template <class T>
std::pair<std::vector<const megcc::KernelGen::KernelFunc*>,
          const megcc::KernelGen::DeduceFunc*>
GetKernels(megcc::KernelGen::Arch platform) {
    llvm::errs() << "no implement yet\n";
    abort();
}

##define INSTANCE_GET_KERNELS(kern_opr, kern_type)                            \
    template <>                                                              \
    std::pair<std::vector<const megcc::KernelGen::KernelFunc*>,              \
              const megcc::KernelGen::DeduceFunc*>                           \
    GetKernels<kern_opr>(megcc::KernelGen::Arch platform) {                  \
        return megcc::KernelGen::KernelPack::GetKernel(kern_type, platform); \
    }

template <class T, typename... Args>
void addBuiltinTemplatesOpr(mlir::Kernel::KernelTemplateRegistry& registry,
                            megcc::KernelGen::Arch arch, Args&&... args) {
    auto kernels = GetKernels<T>(arch);
    for (const auto& kernel : kernels.first) {
        registry.create<T>(kernel, kernels.second, std::forward<Args>(args)...);
    }
}
INSTANCE_GET_KERNELS(mlir::Kernel::Pooling2DKernel, KernType::PoolingKernel)
addBuiltinTemplatesOpr<mlir::Kernel::Pooling2DKernel>(registry, arch);
```
