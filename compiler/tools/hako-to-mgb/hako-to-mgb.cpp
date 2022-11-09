/**
 * \file compiler/tools/hako-to-mgb/hako-to-mgb.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Target/Hako/hako_parse.h"

#include <string.h>
#include <fstream>
#include <memory>

using namespace llvm;

cl::opt<std::string> InputFile(cl::Positional, cl::Required,
                               cl::desc("<input hako model>"));
cl::opt<std::string> OutputFile(cl::Positional, cl::Required,
                                cl::desc("<output mdl file>"));
cl::opt<bool> Verbose("verbose", cl::desc("log more detail information"));
llvm::cl::opt<int> hako_version(
        "hako", llvm::cl::desc("specific megface version used by hako"),
        llvm::cl::init(2));

std::vector<uint8_t> read_file(std::string path) {
    std::vector<uint8_t> res;
    FILE* fin = fopen(path.c_str(), "rb");
    CC_ASSERT(fin) << "can not open " << path << "\n";
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    res.resize(size);
    fseek(fin, 0, SEEK_SET);
    auto nr = fread(res.data(), 1, size, fin);
    CC_ASSERT(nr == size);
    fclose(fin);
    return res;
}

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    cl::ParseCommandLineOptions(argc, argv);
    if (Verbose) {
        megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    }
    auto model_buffer = read_file(InputFile.getValue());
    std::ofstream wf(OutputFile.getValue(), std::ios::out | std::ios::binary);
    auto mdl_result = megcc::parse_hako(model_buffer, hako_version.getValue());
    wf.write((char*)mdl_result.data(), mdl_result.size());
    wf.close();
    return 0;
}