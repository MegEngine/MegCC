
#include "test/kernel/common/dnn_proxy.h"
using namespace megcc::test;
namespace megdnn {
namespace test {

#define EMPTY_CV_DEDUCE(_Opr)                                               \
    template <>                                                             \
    struct DnnOprProxy<_Opr> {                                              \
        static void deduce_layout(_Opr* opr, TensorLayoutArray& layouts) {} \
                                                                            \
        static void exec(_Opr* opr, const TensorNDArray& tensors) {}        \
    };

EMPTY_CV_DEDUCE(CVtranspose);
EMPTY_CV_DEDUCE(CVflip);
}  // namespace test
}  // namespace megdnn

void megdnn::CVtranspose::reformat_layout(CVtranspose* opr,
                                          TensorLayoutArray& layouts) {
    auto src_layout = layouts[0];

    mgb_assert(src_layout.ndim == 4);
    size_t n = 1;
    size_t c = src_layout.shape[3];
    size_t h = src_layout.shape[1];
    size_t w = src_layout.shape[2];
    layouts[1].ndim = 4;
    layouts[1].shape[0] = n;
    layouts[1].shape[1] = w;
    layouts[1].shape[2] = h;
    layouts[1].shape[3] = c;
    layouts[1].init_contiguous_stride();

    layouts[0].shape[0] = n;
    layouts[0].shape[1] = w;
    layouts[0].shape[2] = h;
    layouts[0].shape[3] = c;
    layouts[0].stride[0] = {(ptrdiff_t)(h * w * c)};
    layouts[0].stride[1] = {(ptrdiff_t)c};
    layouts[0].stride[2] = {(ptrdiff_t)(w * c)};
    layouts[0].stride[3] = {(ptrdiff_t)1};
}

void megdnn::CVflip::reformat_layout(CVflip* opr, TensorLayoutArray& layouts) {
    auto src_layout = layouts[0];
    layouts[1] = src_layout;
}
