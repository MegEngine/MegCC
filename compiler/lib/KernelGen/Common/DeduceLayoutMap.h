#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
struct DeduceLayoutMap {
    DeduceLayoutMap();
    std::unordered_map<KernelPack::KernType, std::shared_ptr<DeduceFunc>> map;
};

}  // namespace KernelGen
}  // namespace megcc