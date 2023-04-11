#include "./impl.h"
#include "./best_fit.h"
#include "./interval_move.h"
#include "./pushdown.h"

#include <map>

using namespace mlir::Kernel::migrate;

constexpr size_t StaticMemAllocImplHelper::INVALID;

StaticMemAllocImplHelper::Interval* StaticMemAllocImplHelper::Interval::
        overwrite_dest_root_path_compression() {
    auto&& ptr = m_overwrite_dest_root;
    if (!ptr)
        return this;
    auto root = ptr->overwrite_dest_root_path_compression();
    if (root != ptr) {
        m_offset_in_overwrite_dest_root += ptr->m_offset_in_overwrite_dest_root;
        ptr = root;
    }
    return root;
}

void StaticMemAllocImplHelper::init_overwrite_dest() {
    for (auto&& spec : m_overwrite_spec) {
        auto src = m_interval_storage.data() + std::get<0>(spec),
             dest = m_interval_storage.data() + std::get<1>(spec);
        // src overwrites a part in dest
        size_t offset = std::get<2>(spec);
        CC_ASSERT(src->time_begin < dest->time_end);

        auto orig_src = dest->m_overwrite_src;

        // each interval could only be overwritten by one interval, and we
        // prefer the interval with largest size to be overwritter
        if (src->time_begin == dest->time_end - 1 && !src->m_overwrite_dest &&
            (!orig_src || src->size > orig_src->size)) {
            if (orig_src) {
                orig_src->m_overwrite_dest = nullptr;
                orig_src->m_offset_in_overwrite_dest = 0;
            }
            dest->m_overwrite_src = src;
            src->m_overwrite_dest = dest;
            src->m_offset_in_overwrite_dest = offset;
        }
    }

    for (auto&& i : m_interval_storage) {
        if (i.m_overwrite_dest) {
            i.m_overwrite_dest_root = i.m_overwrite_dest;
            i.m_offset_in_overwrite_dest_root = i.m_offset_in_overwrite_dest;
            CC_ASSERT(i.m_overwrite_dest->m_overwrite_src == &i);
        }
        if (i.m_overwrite_src)
            CC_ASSERT(i.m_overwrite_src->m_overwrite_dest == &i);
    }

    for (auto&& i : m_interval_storage)
        i.overwrite_dest_root_path_compression();
}

size_t StaticMemAllocImplHelper::add(
        size_t begin, size_t end, size_t size, UserKeyType key) {
    CC_ASSERT(begin < end);
    auto id = m_interval_storage.size();
    m_interval_storage.push_back({begin, end, size + m_padding, key, id});
    return id;
}

StaticMemAlloc& StaticMemAllocImplHelper::add_overwrite_spec(
        size_t iid_src, size_t iid_dest, size_t offset) {
    auto &&src = m_interval_storage.at(iid_src),
         &&dest = m_interval_storage.at(iid_dest);
    CC_ASSERT(iid_src != iid_dest);
    CC_ASSERT(offset + src.size <= dest.size);
    m_overwrite_spec.emplace_back(iid_src, iid_dest, offset);
    return *this;
}

size_t StaticMemAllocImplHelper::get_start_addr(UserKeyType key) const {
    return m_userkey2itrv.at(key)->addr_begin;
}

StaticMemAlloc& StaticMemAllocImplHelper::solve() {
    m_interval.clear();
    m_interval.reserve(m_interval_storage.size());
    m_userkey2itrv.clear();
    for (auto&& i : m_interval_storage) {
        m_interval.push_back(&i);
        auto ist = m_userkey2itrv.insert({i.key, &i});
        CC_ASSERT(ist.second) << "duplicated user key";
    }
    init_overwrite_dest();
    do_solve();
    check_result_and_calc_lower_bound();
    return *this;
}

void StaticMemAllocImplHelper::check_result_and_calc_lower_bound() {
    size_t peak = 0;

    // time => pair(alloc, free)
    using TimeEvent = std::pair<IntervalPtrArray, IntervalPtrArray>;
    std::map<size_t, TimeEvent> time2event;

    for (auto&& i : m_interval_storage) {
        CC_ASSERT(i.addr_begin != INVALID);
        time2event[i.time_begin_orig].first.push_back(&i);
        time2event[i.time_end_orig].second.push_back(&i);
        update_max(peak, i.addr_end());
        if (i.is_overwrite_root()) {
            // modify size for calc lower bound
            i.size = align(i.size_orig);
            CC_ASSERT(i.addr_begin == align(i.addr_begin));
        } else {
            auto offset = i.offset_in_overwrite_dest_root();
            i.size = align(offset + i.size_orig) -
                     (offset - (offset & (m_alignment - 1)));
        }
    }
    CC_ASSERT(peak <= tot_alloc() && align(peak) == align(tot_alloc()));

    // get lower bound
    {
        m_peak_lower_bound = 0;
        size_t usage = 0;
        for (auto&& tpair : time2event) {
            for (auto i : tpair.second.first) {
                if (i->is_overwrite_root())
                    usage += i->size;
            }
            for (auto&& i : tpair.second.second) {
                usage -= i->size;
                if (i->m_overwrite_src) {
                    // this interval is overwritten by another one, so count its
                    // size in current usage
                    usage += i->m_overwrite_src->size;
                }
            }
            update_max(m_peak_lower_bound, usage);
        }
        CC_ASSERT(!usage);
    }

    // restore time and size; check overwrite addr
    for (auto&& i : m_interval_storage) {
        i.time_begin = i.time_begin_orig;
        i.time_end = i.time_end_orig;
        i.size = i.size_orig;

        if (!i.is_overwrite_root()) {
            CC_ASSERT(
                    i.overwrite_dest()->addr_begin + i.offset_in_overwrite_dest() ==
                    i.addr_begin);
        }
    }

    std::map<size_t, Interval*> cur_allocated;
    IntervalPtrArray id_overwriter;

    auto remove_alloc = [&](Interval* i) {
        auto iter = cur_allocated.find(i->addr_begin);
        CC_ASSERT(iter != cur_allocated.end() && iter->second == i);
        cur_allocated.erase(iter);

        if (auto s = i->overwrite_src()) {
            auto ins = cur_allocated.insert({s->addr_begin, s});
            CC_ASSERT(ins.second);
        }
    };

    // check for conflicts
    for (auto&& tpair : time2event) {
        // free and set overwriter addr
        id_overwriter.clear();
        for (auto i : tpair.second.second) {
            if (!i->is_overwrite_root() &&
                i->time_end_orig == i->overwrite_dest()->time_end_orig &&
                !i->offset_in_overwrite_dest()) {
                // a overwrites b, a and b share same time end, zero offset
                CC_ASSERT(i->addr_begin == i->overwrite_dest()->addr_begin);
                id_overwriter.push_back(i);
                continue;
            }
            remove_alloc(i);
        }
        for (auto i : id_overwriter)
            remove_alloc(i);

        // alloc
        for (auto i : tpair.second.first) {
            auto iter = cur_allocated.lower_bound(i->addr_begin);

            if (i->is_overwrite_root()) {
                if (iter != cur_allocated.end()) {
                    CC_ASSERT(i->addr_end() <= iter->first);
                }
                if (!cur_allocated.empty() && iter != cur_allocated.begin()) {
                    --iter;
                    CC_ASSERT(iter->second->addr_end() <= i->addr_begin);
                }
                cur_allocated[i->addr_begin] = i;
            }
        }
    }

    CC_ASSERT(cur_allocated.empty());
}

StaticMemAllocImplHelper::~StaticMemAllocImplHelper() noexcept = default;

std::unique_ptr<StaticMemAlloc> StaticMemAlloc::make(AllocatorAlgo algo) {
    switch (algo) {
        case AllocatorAlgo::INTERVAL_MOVE:
            return std::make_unique<StaticMemAllocIntervalMove>();
        case AllocatorAlgo::BEST_FIT:
            return std::make_unique<StaticMemAllocBestFit>();
        case AllocatorAlgo::PUSHDOWN:
            return std::make_unique<StaticMemAllocPushdown>();
        default:
            CC_ASSERT(0) << "unknown mem allocator algorithm";
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
