#pragma once
#include "megdnn/oprs.h"
namespace megdnn {
class CVtranspose {
public:
    struct Param {};
    Param& param() { return m_param; }
    using DnnOpr = megdnn::RelayoutForward;
    DnnOpr::Param dnn_param(Param ori_param) { return DnnOpr::Param(); }
    static void reformat_layout(CVtranspose* opr, TensorLayoutArray& layouts);

private:
    Param m_param;
};

class CVflip {
public:
    using DnnOpr = megdnn::FlipForward;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVflip* opr, TensorLayoutArray& layouts);

private:
    Param m_param;
};

class CVResize {
public:
    using DnnOpr = megdnn::Resize;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVResize* opr, TensorLayoutArray& layouts){};

private:
    Param m_param;
};

class CVRotate {
public:
    using DnnOpr = megdnn::Rotate;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVRotate* opr, TensorLayoutArray& layouts){};

private:
    Param m_param;
};

class CVRoicopy {
public:
    using DnnOpr = megdnn::ROICopy;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVRoicopy* opr, TensorLayoutArray& layouts){};

private:
    Param m_param;
};

class CVCvtColor {
public:
    using DnnOpr = megdnn::CvtColor;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVCvtColor* opr, TensorLayoutArray& layouts){};

private:
    Param m_param;
};

class CVWarpAffine {
public:
    using DnnOpr = megdnn::WarpAffine;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVWarpAffine* opr, TensorLayoutArray& layouts){};

private:
    Param m_param;
};

class CVGaussianBlur {
public:
    using DnnOpr = megdnn::GaussianBlur;
    using Param = DnnOpr::Param;
    Param& param() { return m_param; }

    DnnOpr::Param dnn_param(Param ori_param) { return ori_param; }
    static void reformat_layout(CVGaussianBlur* opr, TensorLayoutArray& layouts){};

private:
    Param m_param;
};

}  // namespace megdnn