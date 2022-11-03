// BEGIN_INTERNAL
//! prebuild with megengine 8.11.1
// BEGIN_INTERNAL
template<>
struct EnumTrait<::megdnn::param::PoolingV0::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::PoolingV0::Mode v) {
        switch (v) {
            case ::megdnn::param::PoolingV0::Mode::MAX : return "MAX";
case ::megdnn::param::PoolingV0::Mode::AVERAGE : return "AVERAGE";
case ::megdnn::param::PoolingV0::Mode::AVERAGE_COUNT_EXCLUDE_PADDING : return "AVERAGE_COUNT_EXCLUDE_PADDING";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Convolution::Format> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Convolution::Format v) {
        switch (v) {
            case ::megdnn::param::Convolution::Format::NCHW : return "NCHW";
case ::megdnn::param::Convolution::Format::NHWC : return "NHWC";
case ::megdnn::param::Convolution::Format::NHWCD4 : return "NHWCD4";
case ::megdnn::param::Convolution::Format::NCHW4 : return "NCHW4";
case ::megdnn::param::Convolution::Format::NCHW8 : return "NCHW8";
case ::megdnn::param::Convolution::Format::NCHW32 : return "NCHW32";
case ::megdnn::param::Convolution::Format::NCHW88 : return "NCHW88";
case ::megdnn::param::Convolution::Format::NCHW44 : return "NCHW44";
case ::megdnn::param::Convolution::Format::NCHW44_DOT : return "NCHW44_DOT";
case ::megdnn::param::Convolution::Format::NCHW4_NCHW32 : return "NCHW4_NCHW32";
case ::megdnn::param::Convolution::Format::NCHW32_NCHW4 : return "NCHW32_NCHW4";
case ::megdnn::param::Convolution::Format::NCHW4_NCHW : return "NCHW4_NCHW";
case ::megdnn::param::Convolution::Format::NHWC_NCHW : return "NHWC_NCHW";
case ::megdnn::param::Convolution::Format::NHWC_NCHW4_IC_SMALL : return "NHWC_NCHW4_IC_SMALL";
case ::megdnn::param::Convolution::Format::NCHW_NCHW4_IC_SMALL : return "NCHW_NCHW4_IC_SMALL";
case ::megdnn::param::Convolution::Format::CHWN4 : return "CHWN4";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Argsort::Order> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Argsort::Order v) {
        switch (v) {
            case ::megdnn::param::Argsort::Order::ASCENDING : return "ASCENDING";
case ::megdnn::param::Argsort::Order::DESCENDING : return "DESCENDING";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ConvBiasV0::NonlineMode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ConvBiasV0::NonlineMode v) {
        switch (v) {
            case ::megdnn::param::ConvBiasV0::NonlineMode::IDENTITY : return "IDENTITY";
case ::megdnn::param::ConvBiasV0::NonlineMode::RELU : return "RELU";
case ::megdnn::param::ConvBiasV0::NonlineMode::SIGMOID : return "SIGMOID";
case ::megdnn::param::ConvBiasV0::NonlineMode::H_SWISH : return "H_SWISH";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ConvolutionV0::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ConvolutionV0::Mode v) {
        switch (v) {
            case ::megdnn::param::ConvolutionV0::Mode::CROSS_CORRELATION : return "CROSS_CORRELATION";
case ::megdnn::param::ConvolutionV0::Mode::CONVOLUTION : return "CONVOLUTION";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ConvolutionV0::Sparse> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ConvolutionV0::Sparse v) {
        switch (v) {
            case ::megdnn::param::ConvolutionV0::Sparse::DENSE : return "DENSE";
case ::megdnn::param::ConvolutionV0::Sparse::GROUP : return "GROUP";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ConvolutionV1::ComputeMode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ConvolutionV1::ComputeMode v) {
        switch (v) {
            case ::megdnn::param::ConvolutionV1::ComputeMode::DEFAULT : return "DEFAULT";
case ::megdnn::param::ConvolutionV1::ComputeMode::FLOAT32 : return "FLOAT32";
            default:
                return {};
        }
    }
};

template<>
struct BitCombinedEnumTrait<::megdnn::param::ExecutionPolicy::Strategy> : public std::true_type {
    static inline std::vector<std::string> nameof(::megdnn::param::ExecutionPolicy::Strategy v) {
        std::vector<std::string> ret;
        if (v & ::megdnn::param::ExecutionPolicy::Strategy::HEURISTIC) { ret.push_back("HEURISTIC"); }
if (v & ::megdnn::param::ExecutionPolicy::Strategy::PROFILE) { ret.push_back("PROFILE"); }
if (v & ::megdnn::param::ExecutionPolicy::Strategy::REPRODUCIBLE) { ret.push_back("REPRODUCIBLE"); }
if (v & ::megdnn::param::ExecutionPolicy::Strategy::OPTIMIZED) { ret.push_back("OPTIMIZED"); }
        return ret;
    }
};

template<>
struct EnumTrait<::megdnn::param::BN::ParamDim> : public std::true_type {
    static inline std::string nameof(::megdnn::param::BN::ParamDim v) {
        switch (v) {
            case ::megdnn::param::BN::ParamDim::DIM_11HW : return "DIM_11HW";
case ::megdnn::param::BN::ParamDim::DIM_1CHW : return "DIM_1CHW";
case ::megdnn::param::BN::ParamDim::DIM_1C11 : return "DIM_1C11";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::BN::FwdMode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::BN::FwdMode v) {
        switch (v) {
            case ::megdnn::param::BN::FwdMode::TRAINING : return "TRAINING";
case ::megdnn::param::BN::FwdMode::INFERENCE : return "INFERENCE";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::MatrixMulV1::ComputeMode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::MatrixMulV1::ComputeMode v) {
        switch (v) {
            case ::megdnn::param::MatrixMulV1::ComputeMode::DEFAULT : return "DEFAULT";
case ::megdnn::param::MatrixMulV1::ComputeMode::FLOAT32 : return "FLOAT32";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::MatrixMul::Format> : public std::true_type {
    static inline std::string nameof(::megdnn::param::MatrixMul::Format v) {
        switch (v) {
            case ::megdnn::param::MatrixMul::Format::DEFAULT : return "DEFAULT";
case ::megdnn::param::MatrixMul::Format::MK4 : return "MK4";
case ::megdnn::param::MatrixMul::Format::MK8 : return "MK8";
case ::megdnn::param::MatrixMul::Format::MK4_DOT : return "MK4_DOT";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::CollectiveComm::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::CollectiveComm::Mode v) {
        switch (v) {
            case ::megdnn::param::CollectiveComm::Mode::REDUCE_SUM : return "REDUCE_SUM";
case ::megdnn::param::CollectiveComm::Mode::BROADCAST : return "BROADCAST";
case ::megdnn::param::CollectiveComm::Mode::ALL_GATHER : return "ALL_GATHER";
case ::megdnn::param::CollectiveComm::Mode::REDUCE_SCATTER_SUM : return "REDUCE_SCATTER_SUM";
case ::megdnn::param::CollectiveComm::Mode::ALL_REDUCE_SUM : return "ALL_REDUCE_SUM";
case ::megdnn::param::CollectiveComm::Mode::ALL_REDUCE_MAX : return "ALL_REDUCE_MAX";
case ::megdnn::param::CollectiveComm::Mode::ALL_REDUCE_MIN : return "ALL_REDUCE_MIN";
case ::megdnn::param::CollectiveComm::Mode::ALL_REDUCE_PROD : return "ALL_REDUCE_PROD";
case ::megdnn::param::CollectiveComm::Mode::GATHER : return "GATHER";
case ::megdnn::param::CollectiveComm::Mode::SCATTER : return "SCATTER";
case ::megdnn::param::CollectiveComm::Mode::ALL_TO_ALL : return "ALL_TO_ALL";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Convolution3D::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Convolution3D::Mode v) {
        switch (v) {
            case ::megdnn::param::Convolution3D::Mode::CROSS_CORRELATION : return "CROSS_CORRELATION";
case ::megdnn::param::Convolution3D::Mode::CONVOLUTION : return "CONVOLUTION";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Convolution3D::Sparse> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Convolution3D::Sparse v) {
        switch (v) {
            case ::megdnn::param::Convolution3D::Sparse::DENSE : return "DENSE";
case ::megdnn::param::Convolution3D::Sparse::GROUP : return "GROUP";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Convolution3D::DataType> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Convolution3D::DataType v) {
        switch (v) {
            case ::megdnn::param::Convolution3D::DataType::FLOAT : return "FLOAT";
case ::megdnn::param::Convolution3D::DataType::FLOAT_IO16xC32 : return "FLOAT_IO16xC32";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Convolution3D::Format> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Convolution3D::Format v) {
        switch (v) {
            case ::megdnn::param::Convolution3D::Format::NCDHW : return "NCDHW";
case ::megdnn::param::Convolution3D::Format::NDHWC : return "NDHWC";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::CvtColor::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::CvtColor::Mode v) {
        switch (v) {
            case ::megdnn::param::CvtColor::Mode::RGB2GRAY : return "RGB2GRAY";
case ::megdnn::param::CvtColor::Mode::RGB2YUV : return "RGB2YUV";
case ::megdnn::param::CvtColor::Mode::YUV2RGB : return "YUV2RGB";
case ::megdnn::param::CvtColor::Mode::GRAY2RGB : return "GRAY2RGB";
case ::megdnn::param::CvtColor::Mode::RGBA2RGB : return "RGBA2RGB";
case ::megdnn::param::CvtColor::Mode::RGBA2BGR : return "RGBA2BGR";
case ::megdnn::param::CvtColor::Mode::RGBA2GRAY : return "RGBA2GRAY";
case ::megdnn::param::CvtColor::Mode::RGB2BGR : return "RGB2BGR";
case ::megdnn::param::CvtColor::Mode::BGR2GRAY : return "BGR2GRAY";
case ::megdnn::param::CvtColor::Mode::BGR2RGB : return "BGR2RGB";
case ::megdnn::param::CvtColor::Mode::YUV2GRAY_NV21 : return "YUV2GRAY_NV21";
case ::megdnn::param::CvtColor::Mode::YUV2RGB_NV21 : return "YUV2RGB_NV21";
case ::megdnn::param::CvtColor::Mode::YUV2BGR_NV21 : return "YUV2BGR_NV21";
case ::megdnn::param::CvtColor::Mode::YUV2GRAY_NV12 : return "YUV2GRAY_NV12";
case ::megdnn::param::CvtColor::Mode::YUV2RGB_NV12 : return "YUV2RGB_NV12";
case ::megdnn::param::CvtColor::Mode::YUV2BGR_NV12 : return "YUV2BGR_NV12";
case ::megdnn::param::CvtColor::Mode::YUV2GRAY_YV12 : return "YUV2GRAY_YV12";
case ::megdnn::param::CvtColor::Mode::YUV2RGB_YV12 : return "YUV2RGB_YV12";
case ::megdnn::param::CvtColor::Mode::YUV2BGR_YV12 : return "YUV2BGR_YV12";
case ::megdnn::param::CvtColor::Mode::YUV2GRAY_YU12 : return "YUV2GRAY_YU12";
case ::megdnn::param::CvtColor::Mode::YUV2RGB_YU12 : return "YUV2RGB_YU12";
case ::megdnn::param::CvtColor::Mode::YUV2BGR_YU12 : return "YUV2BGR_YU12";
case ::megdnn::param::CvtColor::Mode::YCrCb2RGB : return "YCrCb2RGB";
case ::megdnn::param::CvtColor::Mode::YCrCb2BGR : return "YCrCb2BGR";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2RGB_NV21 : return "BT601_YUV2RGB_NV21";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2BGR_NV21 : return "BT601_YUV2BGR_NV21";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2RGB_NV12 : return "BT601_YUV2RGB_NV12";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2BGR_NV12 : return "BT601_YUV2BGR_NV12";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2RGB_YV12 : return "BT601_YUV2RGB_YV12";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2BGR_YV12 : return "BT601_YUV2BGR_YV12";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2RGB_YU12 : return "BT601_YUV2RGB_YU12";
case ::megdnn::param::CvtColor::Mode::BT601_YUV2BGR_YU12 : return "BT601_YUV2BGR_YU12";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Elemwise::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Elemwise::Mode v) {
        switch (v) {
            case ::megdnn::param::Elemwise::Mode::RELU : return "RELU";
case ::megdnn::param::Elemwise::Mode::ABS : return "ABS";
case ::megdnn::param::Elemwise::Mode::ACOS : return "ACOS";
case ::megdnn::param::Elemwise::Mode::ASIN : return "ASIN";
case ::megdnn::param::Elemwise::Mode::CEIL : return "CEIL";
case ::megdnn::param::Elemwise::Mode::COS : return "COS";
case ::megdnn::param::Elemwise::Mode::EXP : return "EXP";
case ::megdnn::param::Elemwise::Mode::EXPM1 : return "EXPM1";
case ::megdnn::param::Elemwise::Mode::FLOOR : return "FLOOR";
case ::megdnn::param::Elemwise::Mode::LOG : return "LOG";
case ::megdnn::param::Elemwise::Mode::LOG1P : return "LOG1P";
case ::megdnn::param::Elemwise::Mode::NEGATE : return "NEGATE";
case ::megdnn::param::Elemwise::Mode::SIGMOID : return "SIGMOID";
case ::megdnn::param::Elemwise::Mode::SIN : return "SIN";
case ::megdnn::param::Elemwise::Mode::TANH : return "TANH";
case ::megdnn::param::Elemwise::Mode::ABS_GRAD : return "ABS_GRAD";
case ::megdnn::param::Elemwise::Mode::ADD : return "ADD";
case ::megdnn::param::Elemwise::Mode::FLOOR_DIV : return "FLOOR_DIV";
case ::megdnn::param::Elemwise::Mode::MAX : return "MAX";
case ::megdnn::param::Elemwise::Mode::MIN : return "MIN";
case ::megdnn::param::Elemwise::Mode::MOD : return "MOD";
case ::megdnn::param::Elemwise::Mode::MUL : return "MUL";
case ::megdnn::param::Elemwise::Mode::POW : return "POW";
case ::megdnn::param::Elemwise::Mode::SIGMOID_GRAD : return "SIGMOID_GRAD";
case ::megdnn::param::Elemwise::Mode::SUB : return "SUB";
case ::megdnn::param::Elemwise::Mode::SWITCH_GT0 : return "SWITCH_GT0";
case ::megdnn::param::Elemwise::Mode::TANH_GRAD : return "TANH_GRAD";
case ::megdnn::param::Elemwise::Mode::TRUE_DIV : return "TRUE_DIV";
case ::megdnn::param::Elemwise::Mode::LOG_SUM_EXP : return "LOG_SUM_EXP";
case ::megdnn::param::Elemwise::Mode::LT : return "LT";
case ::megdnn::param::Elemwise::Mode::LEQ : return "LEQ";
case ::megdnn::param::Elemwise::Mode::EQ : return "EQ";
case ::megdnn::param::Elemwise::Mode::SHL : return "SHL";
case ::megdnn::param::Elemwise::Mode::SHR : return "SHR";
case ::megdnn::param::Elemwise::Mode::COND_LEQ_MOV : return "COND_LEQ_MOV";
case ::megdnn::param::Elemwise::Mode::FUSE_MUL_ADD3 : return "FUSE_MUL_ADD3";
case ::megdnn::param::Elemwise::Mode::FUSE_MUL_ADD4 : return "FUSE_MUL_ADD4";
case ::megdnn::param::Elemwise::Mode::FUSE_ADD_RELU : return "FUSE_ADD_RELU";
case ::megdnn::param::Elemwise::Mode::FUSE_ADD_SIGMOID : return "FUSE_ADD_SIGMOID";
case ::megdnn::param::Elemwise::Mode::FUSE_ADD_TANH : return "FUSE_ADD_TANH";
case ::megdnn::param::Elemwise::Mode::FAST_TANH : return "FAST_TANH";
case ::megdnn::param::Elemwise::Mode::FAST_TANH_GRAD : return "FAST_TANH_GRAD";
case ::megdnn::param::Elemwise::Mode::ROUND : return "ROUND";
case ::megdnn::param::Elemwise::Mode::RMULH : return "RMULH";
case ::megdnn::param::Elemwise::Mode::ATAN2 : return "ATAN2";
case ::megdnn::param::Elemwise::Mode::ERF : return "ERF";
case ::megdnn::param::Elemwise::Mode::ERFINV : return "ERFINV";
case ::megdnn::param::Elemwise::Mode::ERFC : return "ERFC";
case ::megdnn::param::Elemwise::Mode::ERFCINV : return "ERFCINV";
case ::megdnn::param::Elemwise::Mode::H_SWISH : return "H_SWISH";
case ::megdnn::param::Elemwise::Mode::H_SWISH_GRAD : return "H_SWISH_GRAD";
case ::megdnn::param::Elemwise::Mode::FUSE_ADD_H_SWISH : return "FUSE_ADD_H_SWISH";
case ::megdnn::param::Elemwise::Mode::NOT : return "NOT";
case ::megdnn::param::Elemwise::Mode::AND : return "AND";
case ::megdnn::param::Elemwise::Mode::OR : return "OR";
case ::megdnn::param::Elemwise::Mode::XOR : return "XOR";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ElemwiseMultiType::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ElemwiseMultiType::Mode v) {
        switch (v) {
            case ::megdnn::param::ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32 : return "FUSE_MUL_ADD3_INT16x32x32x32";
case ::megdnn::param::ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8 : return "FUSE_MUL_ADD3_IXxF32xF32xI8";
case ::megdnn::param::ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8 : return "ROUND_SHR_SATURATE_IXxI8xI8";
case ::megdnn::param::ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8 : return "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8";
case ::megdnn::param::ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8 : return "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8";
case ::megdnn::param::ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16 : return "ROUND_SHR_SATURATE_IXxI8xI16";
case ::megdnn::param::ElemwiseMultiType::Mode::QADD : return "QADD";
case ::megdnn::param::ElemwiseMultiType::Mode::QFUSE_ADD_RELU : return "QFUSE_ADD_RELU";
case ::megdnn::param::ElemwiseMultiType::Mode::QMUL : return "QMUL";
case ::megdnn::param::ElemwiseMultiType::Mode::QMIN : return "QMIN";
case ::megdnn::param::ElemwiseMultiType::Mode::QMAX : return "QMAX";
case ::megdnn::param::ElemwiseMultiType::Mode::QSUB : return "QSUB";
case ::megdnn::param::ElemwiseMultiType::Mode::QTRUE_DIV : return "QTRUE_DIV";
case ::megdnn::param::ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID : return "QFUSE_ADD_SIGMOID";
case ::megdnn::param::ElemwiseMultiType::Mode::QFUSE_ADD_TANH : return "QFUSE_ADD_TANH";
case ::megdnn::param::ElemwiseMultiType::Mode::QRELU : return "QRELU";
case ::megdnn::param::ElemwiseMultiType::Mode::QABS : return "QABS";
case ::megdnn::param::ElemwiseMultiType::Mode::QSIGMOID : return "QSIGMOID";
case ::megdnn::param::ElemwiseMultiType::Mode::QEXP : return "QEXP";
case ::megdnn::param::ElemwiseMultiType::Mode::QTANH : return "QTANH";
case ::megdnn::param::ElemwiseMultiType::Mode::QFUSE_MUL_ADD3 : return "QFUSE_MUL_ADD3";
case ::megdnn::param::ElemwiseMultiType::Mode::QFAST_TANH : return "QFAST_TANH";
case ::megdnn::param::ElemwiseMultiType::Mode::QNEGATE : return "QNEGATE";
case ::megdnn::param::ElemwiseMultiType::Mode::QACOS : return "QACOS";
case ::megdnn::param::ElemwiseMultiType::Mode::QASIN : return "QASIN";
case ::megdnn::param::ElemwiseMultiType::Mode::QCEIL : return "QCEIL";
case ::megdnn::param::ElemwiseMultiType::Mode::QCOS : return "QCOS";
case ::megdnn::param::ElemwiseMultiType::Mode::QEXPM1 : return "QEXPM1";
case ::megdnn::param::ElemwiseMultiType::Mode::QFLOOR : return "QFLOOR";
case ::megdnn::param::ElemwiseMultiType::Mode::QLOG : return "QLOG";
case ::megdnn::param::ElemwiseMultiType::Mode::QLOG1P : return "QLOG1P";
case ::megdnn::param::ElemwiseMultiType::Mode::QSIN : return "QSIN";
case ::megdnn::param::ElemwiseMultiType::Mode::QROUND : return "QROUND";
case ::megdnn::param::ElemwiseMultiType::Mode::QERF : return "QERF";
case ::megdnn::param::ElemwiseMultiType::Mode::QERFINV : return "QERFINV";
case ::megdnn::param::ElemwiseMultiType::Mode::QERFC : return "QERFC";
case ::megdnn::param::ElemwiseMultiType::Mode::QERFCINV : return "QERFCINV";
case ::megdnn::param::ElemwiseMultiType::Mode::QABS_GRAD : return "QABS_GRAD";
case ::megdnn::param::ElemwiseMultiType::Mode::QFLOOR_DIV : return "QFLOOR_DIV";
case ::megdnn::param::ElemwiseMultiType::Mode::QMOD : return "QMOD";
case ::megdnn::param::ElemwiseMultiType::Mode::QSIGMOID_GRAD : return "QSIGMOID_GRAD";
case ::megdnn::param::ElemwiseMultiType::Mode::QSWITCH_GT0 : return "QSWITCH_GT0";
case ::megdnn::param::ElemwiseMultiType::Mode::QTANH_GRAD : return "QTANH_GRAD";
case ::megdnn::param::ElemwiseMultiType::Mode::QLT : return "QLT";
case ::megdnn::param::ElemwiseMultiType::Mode::QLEQ : return "QLEQ";
case ::megdnn::param::ElemwiseMultiType::Mode::QEQ : return "QEQ";
case ::megdnn::param::ElemwiseMultiType::Mode::QPOW : return "QPOW";
case ::megdnn::param::ElemwiseMultiType::Mode::QLOG_SUM_EXP : return "QLOG_SUM_EXP";
case ::megdnn::param::ElemwiseMultiType::Mode::QFAST_TANH_GRAD : return "QFAST_TANH_GRAD";
case ::megdnn::param::ElemwiseMultiType::Mode::QATAN2 : return "QATAN2";
case ::megdnn::param::ElemwiseMultiType::Mode::QCOND_LEQ_MOV : return "QCOND_LEQ_MOV";
case ::megdnn::param::ElemwiseMultiType::Mode::QH_SWISH : return "QH_SWISH";
case ::megdnn::param::ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH : return "QFUSE_ADD_H_SWISH";
case ::megdnn::param::ElemwiseMultiType::Mode::QH_SWISH_GRAD : return "QH_SWISH_GRAD";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ROIAlignV0::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ROIAlignV0::Mode v) {
        switch (v) {
            case ::megdnn::param::ROIAlignV0::Mode::MAX : return "MAX";
case ::megdnn::param::ROIAlignV0::Mode::AVERAGE : return "AVERAGE";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::ROIPooling::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::ROIPooling::Mode v) {
        switch (v) {
            case ::megdnn::param::ROIPooling::Mode::MAX : return "MAX";
case ::megdnn::param::ROIPooling::Mode::AVERAGE : return "AVERAGE";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Reduce::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Reduce::Mode v) {
        switch (v) {
            case ::megdnn::param::Reduce::Mode::SUM : return "SUM";
case ::megdnn::param::Reduce::Mode::SUM_SQR : return "SUM_SQR";
case ::megdnn::param::Reduce::Mode::PRODUCT : return "PRODUCT";
case ::megdnn::param::Reduce::Mode::MIN : return "MIN";
case ::megdnn::param::Reduce::Mode::MAX : return "MAX";
case ::megdnn::param::Reduce::Mode::MEAN : return "MEAN";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::Reduce::DataType> : public std::true_type {
    static inline std::string nameof(::megdnn::param::Reduce::DataType v) {
        switch (v) {
            case ::megdnn::param::Reduce::DataType::DEFAULT : return "DEFAULT";
case ::megdnn::param::Reduce::DataType::FLOAT_IO16xC32 : return "FLOAT_IO16xC32";
case ::megdnn::param::Reduce::DataType::FLOAT_O32xC32 : return "FLOAT_O32xC32";
case ::megdnn::param::Reduce::DataType::FLOAT_O16xC32 : return "FLOAT_O16xC32";
case ::megdnn::param::Reduce::DataType::QUINT_I8xO32 : return "QUINT_I8xO32";
case ::megdnn::param::Reduce::DataType::QINT_I8xO32 : return "QINT_I8xO32";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::WarpPerspectiveV1::InterpolationMode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::WarpPerspectiveV1::InterpolationMode v) {
        switch (v) {
            case ::megdnn::param::WarpPerspectiveV1::InterpolationMode::NEAREST : return "NEAREST";
case ::megdnn::param::WarpPerspectiveV1::InterpolationMode::LINEAR : return "LINEAR";
case ::megdnn::param::WarpPerspectiveV1::InterpolationMode::AREA : return "AREA";
case ::megdnn::param::WarpPerspectiveV1::InterpolationMode::CUBIC : return "CUBIC";
case ::megdnn::param::WarpPerspectiveV1::InterpolationMode::LANCZOS4 : return "LANCZOS4";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::WarpPerspectiveV1::BorderMode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::WarpPerspectiveV1::BorderMode v) {
        switch (v) {
            case ::megdnn::param::WarpPerspectiveV1::BorderMode::REPLICATE : return "REPLICATE";
case ::megdnn::param::WarpPerspectiveV1::BorderMode::REFLECT : return "REFLECT";
case ::megdnn::param::WarpPerspectiveV1::BorderMode::REFLECT_101 : return "REFLECT_101";
case ::megdnn::param::WarpPerspectiveV1::BorderMode::WRAP : return "WRAP";
case ::megdnn::param::WarpPerspectiveV1::BorderMode::CONSTANT : return "CONSTANT";
case ::megdnn::param::WarpPerspectiveV1::BorderMode::TRANSPARENT : return "TRANSPARENT";
case ::megdnn::param::WarpPerspectiveV1::BorderMode::ISOLATED : return "ISOLATED";
            default:
                return {};
        }
    }
};

template<>
struct EnumTrait<::megdnn::param::TopK::Mode> : public std::true_type {
    static inline std::string nameof(::megdnn::param::TopK::Mode v) {
        switch (v) {
            case ::megdnn::param::TopK::Mode::KTH_ONLY : return "KTH_ONLY";
case ::megdnn::param::TopK::Mode::VALUE_IDX_NOSORT : return "VALUE_IDX_NOSORT";
case ::megdnn::param::TopK::Mode::VALUE_IDX_SORTED : return "VALUE_IDX_SORTED";
            default:
                return {};
        }
    }
};
