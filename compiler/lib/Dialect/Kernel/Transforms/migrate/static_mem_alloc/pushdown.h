/**
 * \file
 * compiler/lib/Dialect/Kernel/Transforms/migrate/static_mem_alloc/pushdown.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#pragma once

#include "./impl.h"

namespace mlir {
namespace Kernel {
namespace migrate {

class StaticMemAllocPushdown final : public StaticMemAllocImplHelper {
    class BestfitPrealloc;

    size_t m_peak_usage = 0;

    /*!
     * intervals that lie directly below this interval; address of each interval
     * is max end address of those in below. Indexed by interval ID
     */
    std::vector<IntervalPtrArray> m_interval_below;

    /*!
     * \brief compute topology order of inervals; result represented in
     *      m_interval_below
     */
    void init_topo_order();

    size_t get_interval_addr_end(Interval* interval);

public:
    void do_solve() override;

    size_t tot_alloc() const override { return m_peak_usage; }
};

}  // namespace migrate
}  // namespace Kernel
}  // namespace mlir

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
