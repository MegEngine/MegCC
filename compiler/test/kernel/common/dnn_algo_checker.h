/**
 * \file
 * compiler/test/kernel/common/dnn_algo_checker.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "./dnn_proxy.h"
#include "./dnn_proxy_algo.h"
#include "megdnn/basic_types.h"

#include <memory>
#include <regex>
#include <unordered_map>

namespace megdnn {
namespace test {

using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;
using TensorLayoutArray = megdnn::SmallVector<megdnn::TensorLayout>;
using TensorShapeArray = megdnn::SmallVector<megdnn::TensorShape>;

struct ExecutionPolicyAlgoName {
    std::string name;
    std::vector<ExecutionPolicyAlgoName> sub_policy_names;

    ExecutionPolicyAlgoName(const char* name) : name{name} {}

    ExecutionPolicyAlgoName(
            const char* name, const std::vector<ExecutionPolicyAlgoName>& sub_policy)
            : name{name}, sub_policy_names{sub_policy} {}
};
/*!
 * \brief a callable to check that given algorithm is used for heuristic
 * \param require_algo if its value is true, then requires
 *      get_algorithm_heuristic() to return the expected algo; otherwise the
 *      expected algo must exist in get_all_algorithms() and it would be set to
 *      be used
 */
template <class Opr, typename OprAlgoProxy = OprAlgoProxy<Opr>>
class AlgoChecker {
public:
    AlgoChecker(ExecutionPolicyAlgoName name, bool* require_algo = nullptr)
            : m_policy_name{name}, m_require_algo{require_algo} {}

    AlgoChecker(ExecutionPolicy policy, bool* require_algo = nullptr)
            : m_policy{policy}, m_require_algo{require_algo} {}

    static ExecutionPolicy construct_execution_policy_from_name(
            const ExecutionPolicyAlgoName& policy_name,
            const TensorLayoutArray& layouts, const std::string& param,
            Handle* handle) {
        ExecutionPolicy ret;
        mgb_assert(layouts.size() == OprTrait<Opr>::arity);
        auto opr = handle->create_operator<Opr>();
        opr->param() = Algorithm::deserialize_read_pod<typename Opr::Param>(param);
        for (auto algo_info :
             AlgoProxy<Opr, OprTrait<Opr>::arity>::get_all_algorithms_info(
                     opr.get(), layouts)) {
            if (std::regex_match(
                        algo_info.desc.name,
                        std::regex("(" + policy_name.name + ")(.*)"))) {
                ret.algo = algo_info.desc;
            } else {
                continue;
            }

            Algorithm* algo = opr->get_algorithm_from_desc(algo_info.desc);
            std::vector<Algorithm::SearchItem>&& sub_items =
                    algo->get_subopr_list(layouts, opr.get());
            if (sub_items.size() != policy_name.sub_policy_names.size()) {
                printf("Invalid sub_policy_names in %s, expected %zu but got "
                       "%zu\n",
                       algo_info.desc.name.c_str(), sub_items.size(),
                       policy_name.sub_policy_names.size());
                return {};
            }
            FOREACH_OPR_TYPE_DISPATCH(sub_items, {
                ExecutionPolicy policy =
                        AlgoChecker<_Opr>::construct_execution_policy_from_name(
                                policy_name.sub_policy_names[_item_idx], _item.layouts,
                                _item.param, handle);
                ret.sub_policy.push_back(policy);
            });
            return ret;
        }
        return ret;
    }

    void operator()(Opr* opr, const TensorNDArray& arr) {
        TensorLayoutArray layouts;
        for (auto&& val : arr) {
            layouts.push_back(val.layout);
        }
        if (!m_policy_name.name.empty()) {
            std::string param_str;
            Algorithm::serialize_write_pod(opr->param(), param_str);
            m_policy = construct_execution_policy_from_name(
                    m_policy_name, layouts, param_str, opr->handle());
            ASSERT_TRUE(m_policy.algo.valid())
                    << "algorithm " << m_policy_name.name << " not found";
        }
        if (m_require_algo && *m_require_algo) {
            auto algo = OprAlgoProxy::get_algorithm_info_heuristic(opr, layouts);
            ASSERT_STREQ(
                    opr->get_algorithm_from_desc(m_policy.algo)->name(),
                    algo.desc.name.c_str());
        } else {
            opr->execution_policy() = m_policy;
        }
        printf("run with dnn algo: %s\n", m_policy.algo.name.c_str());
    }

private:
    ExecutionPolicyAlgoName m_policy_name;
    ExecutionPolicy m_policy;
    bool* m_require_algo;
};

template <typename Opr>
void construct_sub_execution_policy_heuristic(
        ExecutionPolicy& policy, const TensorLayoutArray& layouts,
        const std::string& param, Handle* handle) {
    megdnn_assert(layouts.size() == OprTrait<Opr>::arity);
    auto opr = handle->create_operator<Opr>();
    opr->param() = Algorithm::deserialize_read_pod<typename Opr::Param>(param);
    if (!policy.algo.valid()) {
        policy.algo =
                AlgoProxy<Opr, OprTrait<Opr>::arity>::get_algorithm_info_heuristic(
                        opr.get(), layouts)
                        .desc;
    }

    Algorithm* algo = opr->get_algorithm_from_desc(policy.algo);
    std::vector<Algorithm::SearchItem>&& sub_items =
            algo->get_subopr_list(layouts, opr.get());
    FOREACH_OPR_TYPE_DISPATCH(sub_items, {
        policy.sub_policy.push_back(ExecutionPolicy{});
        construct_sub_execution_policy_heuristic<_Opr>(
                policy.sub_policy.back(), _item.layouts, _item.param, handle);
    });
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
