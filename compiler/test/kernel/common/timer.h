#pragma once
#include <chrono>

namespace megcc {
namespace test {

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

public:
    Timer() { reset(); }
    void reset() {
        m_started = false;
        m_stopped = false;
    }
    void start() {
        mgb_assert(!m_started);
        mgb_assert(!m_stopped);
        m_started = true;
        m_start_point = clock::now();
    }
    void stop() {
        mgb_assert(m_started);
        mgb_assert(!m_stopped);
        m_stopped = true;
        m_stop_point = clock::now();
    }
    size_t get_time_in_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                       m_stop_point - m_start_point)
                .count();
    }

private:
    bool m_started, m_stopped;
    time_point m_start_point, m_stop_point;
};

}  // namespace test
}  // namespace megcc

// vim: syntax=cpp.doxygen
