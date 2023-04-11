#include <gtest/gtest.h>
#include "megcc_test_config.h"
#if MEGCC_TEST_GEN
#include <fstream>
#include <iostream>
#include <string>
#include "megbrain/common.h"
#include "test/kernel/common/target_module.h"
#endif

int main(int argc, char** argv) {
#if MEGCC_TEST_GEN
    printf("arg1 must be gen dir\n");
    mgb_assert(argc >= 1);
    std::string gen_dir(argv[1]);
    argc -= 1;
    argv += 1;
#endif
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

#if MEGCC_TEST_GEN
    auto& c_module = megcc::test::TargetModule::get_global_target_module();
    c_module.write_to_dir(gen_dir);
    printf("generate finish %s\n", gen_dir.c_str());
#endif
    return ret;
}