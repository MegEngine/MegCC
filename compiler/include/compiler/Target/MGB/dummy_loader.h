/**
 * \file compiler/include/compiler/Target/MGB/dummy_loader.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <malloc.h>
#include <cassert>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>
#include "megbrain/serialization/extern_c_opr.h"

namespace {
struct LoaderInfo {
    std::unordered_map<std::string,
                       std::pair<std::vector<std::vector<uint32_t>>,
                                 std::vector<uint32_t>>>
            m_name_2_outputinfo;
    std::unordered_map<std::string, std::string> m_envs;
    std::pair<std::string, std::string> m_loader_path_with_interface;
};
static LoaderInfo loaderInfo;

class MGBOprDescImpl {
    static inline const std::pair<std::vector<std::vector<uint32_t>>,
                                  std::vector<uint32_t>>&
    get_output_info(const std::string& loader_name) {
        const auto& name2outputinfo = loaderInfo.m_name_2_outputinfo;
        auto&& iter = name2outputinfo.find(loader_name);
        if (iter != name2outputinfo.end())
            return iter->second;
        else if (name2outputinfo.size() == 1)
            return name2outputinfo.begin()->second;
        else {
            CC_ASSERT(0)
                    << "Please check loader name in command line args whether "
                       "consistent with loader name in dumped model.\n";
            return {};
        }
    }

    static void release(MGBOprDesc* self) {
        free(self->user_data);
        delete self;
    }

    static size_t hash(const MGBOprDesc* self) { return 1; }

    static int is_same(const MGBOprDesc* self, const MGBOprDesc* rhs) {
        CC_ABORT << "The function 'is_same' is part of the dummy loader, just "
                    "for "
                    "compile but should NOT be called.\n";
        return 1;
    }

    static void execute(const MGBOprDesc* self, const MGBTensor* input,
                        const MGBTensor* output) {
        CC_ABORT << "The function 'execute' is part of the dummy loader, just "
                    "for "
                    "compile but should NOT be called.\n";
    }

    static void infer_shape(const MGBOprDesc* self, const MGBTensorShape* input,
                            MGBTensorShape* output) {
        auto&& output_shapes =
                get_output_info(reinterpret_cast<char*>(self->user_data)).first;
        for (size_t i = 0; i < self->nr_output; ++i) {
            output[i].ndim = output_shapes[i].size();
            for (size_t j = 0; j < output[i].ndim; ++j)
                output[i].shape[j] = output_shapes[i][j];
        }
    }

    static void infer_dtype(const struct MGBOprDesc* self,
                            const MGBDType* input, MGBDType* output) {
        auto&& output_dtypes =
                get_output_info(reinterpret_cast<char*>(self->user_data))
                        .second;
        for (size_t i = 0; i < self->nr_output; ++i)
            output[i] = static_cast<MGBDType>(output_dtypes[i]);
    }

public:
    static MGBOprDesc* make(const std::string& loader_name) {
        auto desc = std::make_unique<MGBOprDesc>();

        uint32_t nr_output = get_output_info(loader_name).first.size();
        mgb_init_opr_desc(desc.get(), nr_output, "dummy");
#define cb(func) desc->func = func;
        MGB_OPR_DESC_FOREACH_MEM_FN(cb)
#undef cb
        desc->infer_dtype = infer_dtype;
        // copy loader name into desc->user_data
        desc->user_data = malloc(loader_name.size() + 1);
        memcpy(desc->user_data, loader_name.c_str(), loader_name.size());
        reinterpret_cast<char*>(desc->user_data)[loader_name.size()] = '\0';

        return desc.release();
    }
};

class MGBOprLoaderImpl {
    static std::map<std::string, void*> user_datas;
    // extra_data format:
    // total_len
    // nr_env
    //     ENV_len_1:ENV_1:VALUE_len_1:VALUE_1
    //     ENV_len_2....
    // loader_path_len:loader_path:interface_len:interface
    static std::shared_ptr<void> extra_data;

    static void make_extra_data() {
        // calculate len
        size_t len = 0;
        size_t nr_env = loaderInfo.m_envs.size();
        len += sizeof(nr_env);  // nr_env
        for (const auto& env : loaderInfo.m_envs) {
            size_t env_len = env.first.size(), value_len = env.second.size();
            len += sizeof(env_len) + env_len + sizeof(value_len) +
                   value_len;  // ENV_len_x + ENV_x + VALUE_len_x + VALUE_x
        }
        len += sizeof(size_t) +
               loaderInfo.m_loader_path_with_interface.first.size() +
               sizeof(size_t) +
               loaderInfo.m_loader_path_with_interface.second
                       .size();  // loader_path_len + loader_path +
                                 // interface_len + interface

        extra_data = std::shared_ptr<void>(malloc(sizeof(size_t) + len), free);
        // fill memory
        void* tmp_p = extra_data.get();
        *(size_t*)(tmp_p) = len;
        tmp_p += sizeof(size_t);
        *(size_t*)tmp_p = nr_env;
        tmp_p += sizeof(size_t);
        for (const auto& env : loaderInfo.m_envs) {
            *(size_t*)tmp_p = env.first.size();
            tmp_p += sizeof(size_t);
            memmove(tmp_p, env.first.c_str(), env.first.size());
            tmp_p += env.first.size();
            *(size_t*)tmp_p = env.second.size();
            tmp_p += sizeof(size_t);
            memmove(tmp_p, env.second.c_str(), env.second.size());
            tmp_p += env.second.size();
        }
        *(size_t*)tmp_p = loaderInfo.m_loader_path_with_interface.first.size();
        tmp_p += sizeof(size_t);
        memmove(tmp_p, loaderInfo.m_loader_path_with_interface.first.c_str(),
                loaderInfo.m_loader_path_with_interface.first.size());
        tmp_p += loaderInfo.m_loader_path_with_interface.first.size();
        *(size_t*)tmp_p = loaderInfo.m_loader_path_with_interface.second.size();
        tmp_p += sizeof(size_t);
        memmove(tmp_p, loaderInfo.m_loader_path_with_interface.second.c_str(),
                loaderInfo.m_loader_path_with_interface.second.size());
    }

    static MGBOprDesc* create_desc(size_t nr_input, const void* buf,
                                   size_t buf_len) {
        std::string name((char*)buf + sizeof(size_t), *(size_t*)buf);
        size_t data_len = buf_len - sizeof(size_t) - *(size_t*)buf;
        void* user_data = malloc(sizeof(size_t) + data_len);
        *(size_t*)(user_data) = data_len;
        memmove(user_data + sizeof(size_t),
                buf + sizeof(size_t) + *(size_t*)buf, data_len);

        user_datas[name] = user_data;

        return MGBOprDescImpl::make(name);
    }

public:
    static std::map<std::string, void*>& get_user_datas() { return user_datas; }
    static void* get_extra_data() { return extra_data.get(); }
    static MGBOprLoader make() {
        make_extra_data();
        return {"extern_opr_dummy", &create_desc};
    }
};
std::map<std::string, void*> MGBOprLoaderImpl::user_datas = {};
std::shared_ptr<void> MGBOprLoaderImpl::extra_data = {};

static void dummy_mgb_c_opr_init(const MGBExternCOprApi* (*get_api)(int)) {
    const MGBExternCOprApi* api = get_api(MGB_EXTERN_C_OPR_VERSION);
    assert(api);
    MGBOprLoader loader = MGBOprLoaderImpl::make();
    api->register_loader(&loader);
}
}  // namespace