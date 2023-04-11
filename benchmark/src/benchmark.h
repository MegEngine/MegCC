#pragma once
#include <string>
#include <vector>
namespace megcc {
namespace Benchmark {
/**
 * Benchmarker interface
 *
 */
class Benchmarker {
public:
    virtual void load_model() = 0;
    virtual void profile() = 0;
    virtual ~Benchmarker() = default;
};
}  // namespace Benchmark

}  // namespace megcc
