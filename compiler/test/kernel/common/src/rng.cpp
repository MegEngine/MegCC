/**
 * \file
 * compiler/test/kernel/common/src/rng.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "test/kernel/common/rng.h"
#include <random>
#include "megdnn/tensor_iter.h"
using namespace megdnn;
using TensorNDArray = SmallVector<TensorND>;
using TensorLayoutArray = SmallVector<TensorLayout>;
using namespace megcc::test;

class RNG::RNGxorshf {
    uint64_t s[2];

public:
    using result_type = uint64_t;

#ifdef WIN32
    static uint64_t min() { return 0; }
    static uint64_t max() { return std::numeric_limits<uint64_t>::max(); }
#else
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() {
        return std::numeric_limits<uint64_t>::max();
    }
#endif

    template <typename T>
    explicit RNGxorshf(T&& gen) {
        s[0] = gen();
        s[1] = gen();
    }

    uint64_t operator()() {
        uint64_t x = s[0];
        uint64_t const y = s[1];
        s[0] = y;
        x ^= x << 23;                          // a
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);  // b, c
        return s[1] + y;
    }
};

class RandomState {
public:
    static std::mt19937_64& generator() { return instance()->m_generator; }

    static void reset() { instance()->m_generator.seed(m_seed); }

private:
    RandomState() : m_generator(m_seed) {}
    std::mt19937_64 m_generator;
    const static int m_seed = 0;
    static RandomState* instance() { return &m_instance; }
    static RandomState m_instance;
};

const int RandomState::m_seed;
RandomState RandomState::m_instance = RandomState();
namespace {
template <typename ctype>
class FastIter {
public:
    FastIter(const TensorND& tensor, bool is_contig) : m_contig(is_contig) {
        if (is_contig) {
            m_ptr = reinterpret_cast<ctype*>(tensor.raw_ptr());
        } else {
            m_naive_iter = tensor_iter_valonly<ctype>(tensor).begin();
        }
    }

    ctype& operator*() {
        if (m_contig) {
            return m_ptr[m_offset];
        } else {
            return *m_naive_iter;
        }
    }

    void operator++() {
        if (m_contig) {
            ++m_offset;
        } else {
            ++m_naive_iter;
        }
    }

private:
    typename TensorIter<ctype, true>::Iter m_naive_iter;
    bool m_contig{false};
    size_t m_offset{0};
    ctype* m_ptr{nullptr};
};
}  // namespace
static void write_helper(const TensorND& tensor, std::function<float()> gen) {
    auto t_dtype = tensor.layout.dtype;

    if (t_dtype.enumv() == DTypeEnum::Float32) {
        FastIter<float> iter(tensor, tensor.layout.is_contiguous());
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            *iter = gen();
            ++iter;
        }
    } else if (t_dtype.enumv() == DTypeEnum::Uint8 ||
               t_dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        auto iter_tensor = tensor;
        iter_tensor.layout.dtype = dtype::Uint8();
        FastIter<uint8_t> iter(iter_tensor, tensor.layout.is_contiguous());
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            int val = (int)gen();
            val = val > 255 ? 255 : val;
            val = val < 0 ? 0 : val;
            *iter = (uint8_t)val;
            ++iter;
        }
    } else if (t_dtype.enumv() == DTypeEnum::Int8 ||
               t_dtype.enumv() == DTypeEnum::QuantizedS8) {
        auto iter_tensor = tensor;
        iter_tensor.layout.dtype = dtype::Int8();
        FastIter<int8_t> iter(iter_tensor, tensor.layout.is_contiguous());
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            int val = (int)gen();
            val = val > 127 ? 127 : val;
            val = val < -128 ? -128 : val;
            *iter = (int8_t)val;
            ++iter;
        }
    } else if (t_dtype.enumv() == DTypeEnum::Int32 ||
               t_dtype.enumv() == DTypeEnum::QuantizedS32) {
        auto iter_tensor = tensor;
        iter_tensor.layout.dtype = dtype::Int32();
        FastIter<int> iter(iter_tensor, tensor.layout.is_contiguous());
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            *iter = gen();
            ++iter;
        }
    } else if (t_dtype.enumv() == DTypeEnum::Int16) {
        auto iter_tensor = tensor;
        iter_tensor.layout.dtype = dtype::Int16();
        FastIter<int16_t> iter(iter_tensor, tensor.layout.is_contiguous());
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            *iter = gen();
            ++iter;
        }
    } else if (t_dtype.enumv() == DTypeEnum::Float16) {
        FastIter<dt_float16> iter(tensor, tensor.layout.is_contiguous());
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            *iter = gen();
            ++iter;
        }
    } else {
        mgb_assert(0, "write_helper not support dtype %s", t_dtype.name());
    }
}

void megcc::test::NormalRNG::gen(const TensorND& tensor) {
    RNGxorshf gen{RandomState::generator()};
    auto gen_func = [&]() { return m_dist(gen); };
    write_helper(tensor, gen_func);
}

void megcc::test::WarpPerspectiveMatRNG::gen(const TensorND& tensor) {
    mgb_assert(tensor.layout.dtype.enumv() == DTypeEnum::Float32);
    mgb_assert(tensor.layout.is_contiguous());
    RNGxorshf gen{RandomState::generator()};
    float* ptr = tensor.ptr<float>();
    for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
        size_t idx = i % 9;
        switch (idx) {
            case 6:
            case 7:
                ptr[i] = m_6_7_dist(gen);
                break;
            case 8:
                ptr[i] = m_8_dist(gen);
                break;

            default:
                ptr[i] = m_default_dist(gen);
                break;
        }
    }
}

void megcc::test::UniformRNG::gen(const TensorND& tensor) {
    mgb_assert(tensor.layout.is_contiguous());
    if (tensor.layout.dtype.enumv() == DTypeEnum::Float32) {
        RNGxorshf gen{RandomState::generator()};
        float* ptr = tensor.ptr<float>();
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            ptr[i] = m_dist(gen);
        }
    } else if (tensor.layout.dtype.enumv() == DTypeEnum::Float16) {
        RNGxorshf gen{RandomState::generator()};
        dt_float16* ptr = tensor.ptr<dt_float16>();
        for (size_t i = 0; i < tensor.layout.total_nr_elems(); ++i) {
            ptr[i] = m_dist(gen);
        }
    } else {
        mgb_assert(0, "UniformRNG not support dtype %s",
                   tensor.layout.dtype.name());
    }
}

void megcc::test::ListRNG::gen(const TensorND& tensor) {
    mgb_assert(tensor.layout.is_contiguous());
    auto gen_func = [&]() {
        if (m_cnt >= m_val.size())
            m_cnt = 0;
        return m_val[m_cnt++];
    };
    write_helper(tensor, gen_func);
}

void megcc::test::UniformIntRNG::gen(const TensorND& tensor) {
    RNGxorshf gen{RandomState::generator()};
    auto gen_func = [&]() { return m_dist(gen); };
    write_helper(tensor, gen_func);
}

void megcc::test::ConstRNG::gen(const megdnn::TensorND& tensor) {
    mgb_assert(tensor.layout.is_contiguous());
    auto gen_func = [=]() { return m_value; };
    write_helper(tensor, gen_func);
}

void megcc::test::SequenceRNG::gen(const megdnn::TensorND& tensor) {
    RNGxorshf gen{RandomState::generator()};
    auto gen_func = [&]() { return m_count++; };
    write_helper(tensor, gen_func);
}
void megcc::test::SequenceRNG::reset() {
    m_count = 0;
}

InvertibleMatrixRNG::InvertibleMatrixRNG()
        : m_rng{new RNGxorshf{RandomState::generator()}} {}

InvertibleMatrixRNG::~InvertibleMatrixRNG() noexcept = default;

template <typename ctype>
void InvertibleMatrixRNG::do_gen(ctype* ptr, size_t batch, size_t n) {
    auto&& gen = *m_rng;
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    for (size_t i = 0; i < batch; ++i, ptr += n * n) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                ptr[j * n + k] = static_cast<ctype>(
                        gen() / (RNGxorshf::max() + 1.0) * 2 - 0.5);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            auto idx = gen() % (n - i) + i;
            ptr[i * n + perm[idx]] +=
                    static_cast<ctype>(gen() / (RNGxorshf::max() + 1.0) + 3);
            std::swap(perm[idx], perm[i]);
        }
    }
}

void InvertibleMatrixRNG::gen(const megdnn::TensorND& tensor) {
#define cb(DType)                                               \
    if (tensor.layout.dtype == DType()) {                       \
        using ctype = typename DTypeTrait<DType>::ctype;        \
        auto ptr = tensor.ptr<ctype>();                         \
        mgb_assert(tensor.layout.ndim >= 2 &&                   \
                   tensor.layout.is_physical_contiguous());     \
        size_t batch = 1;                                       \
        for (size_t i = 0; i < tensor.layout.ndim - 2; ++i) {   \
            batch *= tensor.layout[i];                          \
        }                                                       \
        size_t n = tensor.layout[tensor.layout.ndim - 1];       \
        mgb_assert(n == tensor.layout[tensor.layout.ndim - 2]); \
        do_gen<ctype>(ptr, batch, n);                           \
        return;                                                 \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}