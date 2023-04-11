#include "Activation.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

std::shared_ptr<ActivationGenIntrinsicBase> megcc::KernelGen::GeneralIntrinsic::
        create_activation_gener_instrinsic(std::string mode, std::string ctype) {
    if (ctype == "f32") {
        if (mode == "IDENTITY") {
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::IDENTITY, Dtype::FLOAT32>>();
        } else if (mode == "H_SWISH") {
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::H_SWISH, Dtype::FLOAT32>>();
        } else if (mode == "RELU") {
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::RELU, Dtype::FLOAT32>>();
        } else if (mode == "SIGMOID") {
            //! SIGMOID should impl after matmul
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::IDENTITY, Dtype::FLOAT32>>();
        } else {
            CC_ABORT << "unsupported NonlineMode\n";
            return nullptr;
        }
    } else if (ctype == "f16") {
        if (mode == "IDENTITY") {
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::IDENTITY, Dtype::FLOAT16>>();
        } else if (mode == "H_SWISH") {
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::H_SWISH, Dtype::FLOAT16>>();
        } else if (mode == "RELU") {
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::RELU, Dtype::FLOAT16>>();
        } else if (mode == "SIGMOID") {
            //! SIGMOID should impl after matmul
            return std::make_shared<
                    ActivationGenIntrinsic<NonlineMode::IDENTITY, Dtype::FLOAT16>>();
        } else {
            CC_ABORT << "unsupported NonlineMode\n";
            return nullptr;
        }
    } else {
        CC_ABORT << "unsupported compute dtype" << ctype << "\n";
        return nullptr;
    }
}

// vim: syntax=cpp.doxygen
