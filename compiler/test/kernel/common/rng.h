/**
 * \file
 * compiler/test/kernel/common/rng.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#pragma once
#include <random>
#include "megbrain/common.h"
#include "megdnn/oprs/nn.h"
namespace megcc {
namespace test {
class RNG {
protected:
    class RNGxorshf;

public:
    virtual void gen(const megdnn::TensorND& tensor) = 0;
    virtual void reset(){};
    virtual ~RNG() = default;
};
class NormalRNG final : public RNG {
public:
    NormalRNG(float mean = 0.0f, float stddev = 1.0f) : m_dist(mean, stddev) {}
    void gen(const megdnn::TensorND& tensor) override;

private:
    std::normal_distribution<float> m_dist;
};

class ListRNG final : public RNG {
public:
    ListRNG(std::vector<float> val) : m_val(val) {}
    void gen(const megdnn::TensorND& tensor) override;

private:
    size_t m_cnt{0};
    std::vector<float> m_val;
};

class WarpPerspectiveMatRNG final : public RNG {
public:
    WarpPerspectiveMatRNG()
            : m_default_dist(0, 1), m_6_7_dist(0, 0.01), m_8_dist(1, 0.1) {}
    void gen(const megdnn::TensorND& tensor) override;

private:
    std::normal_distribution<float> m_default_dist;
    std::normal_distribution<float> m_6_7_dist;
    std::normal_distribution<float> m_8_dist;
};

class ConstRNG final : public RNG {
public:
    ConstRNG(float value) : m_value(value) {}
    void gen(const megdnn::TensorND& tensor) override;

private:
    float m_value;
};

class UniformRNG final : public RNG {
public:
    UniformRNG(float start = 0.0f, float end = 1.0f) : m_dist(start, end) {}
    void gen(const megdnn::TensorND& tensor) override;

private:
    std::uniform_real_distribution<float> m_dist;
};

class UniformIntRNG final : public RNG {
public:
    UniformIntRNG(int start = 0, int end = 10) : m_dist(start, end) {}
    void gen(const megdnn::TensorND& tensor) override;

private:
    std::uniform_int_distribution<int32_t> m_dist;
};

class SequenceRNG final : public RNG {
public:
    SequenceRNG() {}
    void gen(const megdnn::TensorND& tensor) override;
    void reset() override;

private:
    int m_count{0};
};

class InvertibleMatrixRNG final : public RNG {
    std::unique_ptr<RNGxorshf> m_rng;

public:
    InvertibleMatrixRNG();
    ~InvertibleMatrixRNG() noexcept;

    void gen(const megdnn::TensorND& tensor) override;

private:
    template <typename ctype>
    void do_gen(ctype* ptr, size_t batch, size_t n);
};

class Float16PeriodicalRNG : public RNG {
public:
    Float16PeriodicalRNG();
    Float16PeriodicalRNG(size_t range);

    void gen(const megdnn::TensorND& tensor) override;
    megdnn::dt_float16 get_single_val();

private:
    void gen_all_valid_float16();
    size_t m_offset;
    std::vector<megdnn::dt_float16> m_sequence;
};

}  // namespace test
}  // namespace megcc