#pragma once
#include <memory>
#include <string>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/Armv7/KernelCommon.h"

namespace megcc {
namespace KernelGen {
namespace Armv7 {
class Int8DirectNchwHelperBase {
public:
    static std::string gen_common_code();
    static std::string gen_copy_padding_code(TContext* ctx);
    static std::string gen_res_store_code(
            const std::string& reg_name, const std::string& dst_name,
            const ArmCommon::ActivationGenIntrinsicBase& act);
    virtual std::string gen_need_copy_padding() const = 0;
    virtual std::string gen_get_rectified_size(TContext* ctx) const = 0;
    std::string gen_do_conv_code(
            TContext* ctx, bool with_bias, std::string nonline_mode);
    virtual ~Int8DirectNchwHelperBase() {}

private:
    static std::string gen_kern_name(TContext* ctx);
    virtual std::string gen_kern(
            TContext* ctx, bool with_bias, std::string nonline_mode,
            std::string func_name) const = 0;
};

class Int8DirectNchwBase : public Armv7ConvImpl {
public:
    Int8DirectNchwBase(Int8DirectNchwHelperBase* helper) : helper(helper) {}
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;

protected:
    std::unique_ptr<Int8DirectNchwHelperBase> helper;
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
