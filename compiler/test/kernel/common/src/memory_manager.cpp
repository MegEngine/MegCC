#include "test/kernel/common/memory_manager.h"
#include "test/kernel/common/dnn_helper.h"
namespace {

using namespace megdnn;
using namespace std;
std::unique_ptr<MemoryManager> create_memory_manager_from_handle(Handle* handle) {
    return make_unique<HandleMemoryManager>(handle);
}

}  // anonymous namespace

megdnn::MemoryManagerHolder megdnn::MemoryManagerHolder::m_instance;

megdnn::HandleMemoryManager::HandleMemoryManager(Handle* handle)
        : MemoryManager(), m_handle(handle) {}

void* megdnn::HandleMemoryManager::malloc(size_t size) {
    auto comp_handle = m_handle->megcore_computing_handle();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreGetDeviceHandle(comp_handle, &dev_handle));
    void* ptr;
    megcore_check(megcoreMalloc(dev_handle, &ptr, size));
    return ptr;
}

void megdnn::HandleMemoryManager::free(void* ptr) {
    auto comp_handle = m_handle->megcore_computing_handle();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreGetDeviceHandle(comp_handle, &dev_handle));
    megcore_check(megcoreFree(dev_handle, ptr));
}

megdnn::MemoryManager* megdnn::MemoryManagerHolder::get(Handle* handle) {
    std::lock_guard<std::mutex> lock(m_map_mutex);
    auto i = m_map.find(handle);
    if (i != m_map.end()) {
        // found
        return i->second.get();
    } else {
        // not found. create it
        auto mm = create_memory_manager_from_handle(handle);
        auto res = mm.get();
        m_map.emplace(std::make_pair(handle, std::move(mm)));
        return res;
    }
}

void MemoryManagerHolder::update(
        Handle* handle, std::unique_ptr<MemoryManager> memory_manager) {
    std::lock_guard<std::mutex> lock(m_map_mutex);
    m_map[handle] = std::move(memory_manager);
}

void MemoryManagerHolder::clear() {
    std::lock_guard<std::mutex> lock(m_map_mutex);
    m_map.clear();
}

// vim: syntax=cpp.doxygen
