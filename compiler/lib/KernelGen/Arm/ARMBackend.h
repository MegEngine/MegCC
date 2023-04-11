#pragma once
namespace megcc {
namespace KernelGen {

struct ARMBackend {
    static constexpr int cacheline_byte = 64;
};

}  // namespace KernelGen
}  // namespace megcc