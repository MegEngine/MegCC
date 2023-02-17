/**
 * \file compiler/tools/mgb-tblgen/mgb-tblgen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "./helper.h"
#include "compiler/Common/Version.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;
using namespace mlir::tblgen;

enum ActionType { None, EnumReflection };

// NOLINTNEXTLINE
llvm::cl::opt<ActionType> action(
        llvm::cl::desc("Action to perform:"),
        llvm::cl::values(clEnumValN(
                EnumReflection, "gen-enum-reflection",
                "Generate enumerate class reflection")));

bool gen_enum_reflection(raw_ostream& os, RecordKeeper& keeper) {
    std::unordered_set<unsigned int> enums;
    std::function<void(const MgbEnumAttr&)> insert;
    insert = [&](const MgbEnumAttr& attr) {
        if (auto alias = llvm::dyn_cast<MgbAliasAttr>(&attr)) {
            auto base = alias->getAliasBase();
            return insert(llvm::cast<MgbEnumAttr>(base));
        }
        if (enums.insert(attr.getBaseRecord()->getID()).second) {
            FmtContext ctx;
            ctx.addSubst(
                    "enumClass", attr.getParentNamespace() + "::" + attr.getEnumName());
            auto addBody = [&](const char* tpl) {
                ctx.addSubst(
                        "body", llvm::join(
                                        llvm::map_range(
                                                attr.getEnumMembers(),
                                                [&](auto&& i) -> std::string {
                                                    return tgfmt(tpl, &ctx, i);
                                                }),
                                        "\n"));
            };
            if (attr.getEnumCombinedFlag()) {
                const char* ifTemp =
                        "if (v & $enumClass::$0) { ret.push_back(\"$0\"); }";
                addBody(ifTemp);
                // clang-format off
                os << tgfmt(R"(
template<>
struct BitCombinedEnumTrait<$enumClass> : public std::true_type {
    static inline std::vector<std::string> nameof($enumClass v) {
        std::vector<std::string> ret;
        $body
        return ret;
    }
};
)", &ctx);
                // clang-format on
            } else {
                const char* caseTemp = "case $enumClass::$0 : return \"$0\";";
                addBody(caseTemp);
                // clang-format off
                os << tgfmt(R"(
template<>
struct EnumTrait<$enumClass> : public std::true_type {
    static inline std::string nameof($enumClass v) {
        switch (v) {
            $body
            default:
                return {};
        }
    }
};
)", &ctx);
                // clang-format on
            }
        }
    };
    foreach_operator(keeper, [&](MgbOp& op) {
        for (auto&& i : op.getAttributes()) {
            if (auto* attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
                insert(*attr);
            }
        }
    });
    return false;
}

int main(int argc, char** argv) {
    llvm::cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv);
    if (action == ActionType::EnumReflection) {
        return TableGenMain(argv[0], &gen_enum_reflection);
    }
    return -1;
}
