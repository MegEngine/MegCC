#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "Arm/ArmCommon/Activation.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {

struct ActivationGenAsmBase {
    virtual std::string GenAsmFloat(
            std::vector<std::string> regs, std::vector<std::string> help_regs) = 0;
    virtual std::string GenAsmInt8(
            std::vector<std::string> regs, std::vector<std::string> help_regs) = 0;
    virtual std::string GenAsmFloat(std::vector<std::string> regs) {
        CC_ASSERT(regs.size() >= 2);
        auto helper_reg = regs.back();
        regs.pop_back();
        return GenAsmFloat(regs, {helper_reg});
    }
    virtual std::string GenAsmQuantInit(
            const std::vector<std::string> args_reg, const std::string& mode,
            const std::vector<std::string> args_ptr);
    virtual std::string GenAsmQuantStore(
            std::vector<std::string> int_regs, std::string src_scale_reg,
            std::string dst_scale_ptr, std::string src_scale_ptr,
            const std::string& output_sym, const int elem_offset,
            const std::string dst_specifier, const std::vector<std::string> args_reg,
            const std::string& mode, bool with_store = true);
};

template <ArmCommon::NonlineMode mode>
struct ActivationGenAsm : public ActivationGenAsmBase {
public:
    std::string GenAsmFloat(
            std::vector<std::string>, std::vector<std::string> help_regs) override {
        return "";
    }
    std::string GenAsmInt8(
            std::vector<std::string>, std::vector<std::string> help_regs) override {
        return "";
    }
};

template <>
struct ActivationGenAsm<ArmCommon::NonlineMode::RELU> : public ActivationGenAsmBase {
public:
    //! the first register is the zero register
    std::string GenAsmFloat(
            std::vector<std::string> registers,
            std::vector<std::string> help_regs) override {
        std::string r_zero = help_regs[0] + ".4s \\n\"\n";
        std::stringstream writer;
        writer << "\n";
        for (size_t i = 0; i < registers.size(); i++) {
            writer << "\"fmax " << registers[i] << ".4s, " << registers[i] << ".4s, "
                   << r_zero;
        }
        return writer.str();
    }
    std::string GenAsmInt8(
            std::vector<std::string> registers,
            std::vector<std::string> help_regs) override {
        // TODO: to add the activation
        CC_ASSERT(0) << "not support asm int8 activate relu";
        return "";
    }
};

std::shared_ptr<ActivationGenAsmBase> create_activation_gener(std::string mode);

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
