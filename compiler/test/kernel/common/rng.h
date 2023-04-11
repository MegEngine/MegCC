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

class ResizeMatRNG : public RNG {
    void gen(const megdnn::TensorND& tensor_) override {
        auto& gen = RandomState::generator();
        std::uniform_real_distribution<megdnn::dt_float32> pdist3(1.9f, 3.1f);
        std::uniform_real_distribution<megdnn::dt_float32> pdist(0.9f, 1.1f);
        std::uniform_real_distribution<megdnn::dt_float32> pdisth(0.4f, 0.6f);
        std::uniform_real_distribution<megdnn::dt_float32> ndist(-1.1f, -0.9f);
        std::uniform_real_distribution<megdnn::dt_float32> ndist3(-3.1f, -1.9f);
        std::uniform_real_distribution<megdnn::dt_float32> ndisth(-0.6f, -0.4f);
        std::uniform_int_distribution<int> dice(0, 5);
        float* ptr = tensor_.ptr<megdnn::dt_float32>();
        auto N = tensor_.layout.shape[0];
        for (size_t n = 0; n < N; ++n) {
            for (size_t i = 0; i < 9; ++i) {
                switch (dice(gen)) {
                    case 0:
                        ptr[i] = pdist3(gen);
                        break;
                    case 1:
                        ptr[i] = pdist(gen);
                        break;
                    case 2:
                        ptr[i] = pdisth(gen);
                        break;
                    case 3:
                        ptr[i] = ndist(gen);
                        break;
                    case 4:
                        ptr[i] = ndist3(gen);
                        break;
                    case 5:
                        ptr[i] = ndisth(gen);
                        break;
                }
            }
            ptr[1] = 0;
            ptr[3] = 0;
            ptr[6] = ptr[7] = 0;
            ptr += 9;
        }
    }
};

}  // namespace test
}  // namespace megcc