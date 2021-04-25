/**
 * \file
 * compiler/lib/Dialect/Kernel/Transforms/migrate/static_mem_alloc/best_fit.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "./best_fit.h"
#include "./best_fit_helper.h"

using namespace mlir::Kernel::migrate;

void StaticMemAllocBestFit::do_solve() {
    BestFitHelper helper;
    helper.alloc = [this](Interval* p) {
        p->addr_begin = this->alloc_aligned_addr(p->size);
    };
    helper.alloc_overwrite = [this](Interval* dest, size_t offset,
                                    Interval* p) {
        this->free(dest->addr_begin);
        auto addr = dest->addr_begin + offset;
        this->alloc_placement(addr, p->size);
        p->addr_begin = addr;
    };
    helper.free = [this](Interval* p) { this->free(p->addr_begin); };
    helper.run(m_interval);
}

size_t StaticMemAllocBestFit::alloc_aligned_addr(size_t size) {
    auto iter = m_free_by_size_addr_align.lower_bound({0, size});
    if (iter == m_free_by_size_addr_align.end()) {
        auto rst = m_top;
        m_top += size;
        auto atop = align(m_top);
        insert_free({m_top, atop - m_top});
        m_top = atop;
        m_allocated_chunk[rst] = size;
        return rst;
    }
    auto aiter = iter->aiter();
    auto alloc_addr = iter->addr_aligned, chk_addr = aiter->first,
         chk_size = aiter->second.size, offset = alloc_addr - chk_addr;
    remove_free_by_aiter(aiter);
    CC_ASSERT(align(chk_addr) == alloc_addr && size + offset <= chk_size);
    insert_free({chk_addr, offset});
    insert_free({alloc_addr + size, chk_size - size - offset});
    m_allocated_chunk[alloc_addr] = size;
    return alloc_addr;
}

void StaticMemAllocBestFit::alloc_placement(size_t addr, size_t size) {
    auto iter = m_free_by_addr.upper_bound(addr);
    --iter;
    auto chk_addr = iter->first, chk_size = iter->second.size;
    CC_ASSERT(chk_addr <= addr && chk_addr + chk_size >= addr + size);
    remove_free_by_aiter(iter);
    insert_free({chk_addr, addr - chk_addr});
    insert_free({addr + size, chk_addr + chk_size - (addr + size)});
    m_allocated_chunk[addr] = size;
}

void StaticMemAllocBestFit::free(size_t addr) {
    auto iter = m_allocated_chunk.find(addr);
    CC_ASSERT(iter != m_allocated_chunk.end());
    merge_free_and_insert({addr, iter->second});
    m_allocated_chunk.erase(iter);
}

void StaticMemAllocBestFit::merge_free_and_insert(Chunk chk) {
    auto iter = m_free_by_addr.lower_bound(chk.addr);

    // merge with prev
    if (iter != m_free_by_addr.begin()) {
        auto iprev = iter;
        --iprev;
        if (iprev->second.size + iprev->first == chk.addr) {
            chk.addr = iprev->first;
            chk.size += iprev->second.size;
            remove_free_by_aiter(iprev);
        }
    }

    // merge with next
    if (iter != m_free_by_addr.end()) {
        if (iter->first == chk.addr_end()) {
            chk.size += iter->second.size;
            remove_free_by_aiter(iter);
        }
    }

    insert_free(chk);
}

void StaticMemAllocBestFit::remove_free_by_aiter(FreeByAddrIter aiter) {
    auto siter = aiter->second.siter;
    if (siter != m_free_by_size_addr_align.end())
        m_free_by_size_addr_align.erase(siter);
    m_free_by_addr.erase(aiter);
}

void StaticMemAllocBestFit::insert_free(const Chunk& chk) {
    if (!chk.size)
        return;

    auto addr_align = align(chk.addr), offset = addr_align - chk.addr;
    size_t size_align = 0;
    if (offset < chk.size)
        size_align = chk.size - offset;

    auto ins0 = m_free_by_addr.insert({chk.addr, FreeBlockByAddr{chk}});
    CC_ASSERT(ins0.second);
    if (size_align) {
        auto ins1 = m_free_by_size_addr_align.insert({addr_align, size_align});
        CC_ASSERT(ins1.second);
        ins0.first->second.siter = ins1.first;
        const_cast<FreeBlockBySizeAddrAligned&>(*ins1.first).aiter() =
                ins0.first;
    } else {
        ins0.first->second.siter = m_free_by_size_addr_align.end();
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
