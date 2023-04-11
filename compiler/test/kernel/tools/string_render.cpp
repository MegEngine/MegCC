#include <regex>
#include <sstream>
#include "Utils/StringTemplate.h"
#include "test/kernel/common/checker.h"

using namespace megcc::KernelGen;

TEST(TOOLS, RenderStr) {
    std::string temp_str = R"(
        int haha = ${val0};
            ${val1}wawa;
                ${val2};kaka
    )";
    std::string target_str = R"(
        int haha = 1234;
            5678wawa;
                9;kaka
    )";
    StringTemplate::KvMap kv_map;
    kv_map["val0"] = "1234";
    kv_map["val1"] = "5678";
    kv_map["val2"] = "9";
    auto res = StringTemplate::render(temp_str, kv_map);
    EXPECT_TRUE(res == target_str);
}

TEST(TOOLS, RenderStrFunc) {
    std::string temp_str = R"(
        int val_0 = 1;
        int val_1 = ${var_0};
        int haha = ${add(val_0, 2, val_1)};
        int wawa = ${gen()};
    )";
    std::string target_str = R"(
        int val_0 = 1;
        int val_1 = 3;
        int haha = val_0 + 2 + val_1;
        int wawa = 1234;
    )";

    StringTemplate::FuncMap func_map;
    func_map["add"] = [](const std::vector<std::string>& str_vec) -> std::string {
        std::stringstream ss;
        for (size_t i = 0; i < str_vec.size(); ++i) {
            ss << str_vec[i];
            if (i != str_vec.size() - 1) {
                ss << " + ";
            }
        }
        return ss.str();
    };
    func_map["gen"] = [](const std::vector<std::string>&) -> std::string {
        return "1234";
    };
    StringTemplate::KvMap kv_map;
    kv_map["var_0"] = "3";
    auto res = StringTemplate::render(temp_str, kv_map, func_map);
    EXPECT_TRUE(res == target_str);
}

TEST(TOOLS, RenderStrFuncStream) {
    std::string temp_str = R"(
        int val_0 = 1;
        int val_1 = ${var_0};
        int haha = ${add(val_0, 2, val_1)};
        int wawa = ${gen()};
        ${compute_vec(dst_v[3], &src_v[0][3], &kernel[6])};
    )";
    std::string target_str = R"(
        int val_0 = 1;
        int val_1 = 3;
        int haha = val_0 + 2 + val_1;
        int wawa = 1234;
        dst_v[3]=&src_v[0][3]+&kernel[6];
    )";
    auto res = StringTemplate::StringTemplateArgs()
                       .add("var_0", 3)
                       .add("add",
                            [](const std::vector<std::string>& str_vec) -> std::string {
                                std::stringstream ss;
                                for (size_t i = 0; i < str_vec.size(); ++i) {
                                    ss << str_vec[i];
                                    if (i != str_vec.size() - 1) {
                                        ss << " + ";
                                    }
                                }
                                return ss.str();
                            })
                       .add("gen",
                            [](const std::vector<std::string>&) -> std::string {
                                return "1234";
                            })
                       .add("compute_vec",
                            [](const std::string& dst, const std::string& src_0,
                               const std::string& src_1) -> std::string {
                                std::stringstream ss;
                                ss << dst << "=" << src_0 << "+" << src_1;
                                return ss.str();
                            })
                       .render(temp_str);
    EXPECT_TRUE(res == target_str);
}

TEST(TOOLS, RenderStrFor) {
    std::string temp_str = "${_unroll(iter, sub_str, stride:1)}";
    auto generted_str =
            StringTemplate::StringTemplateArgs()
                    .add("iter", 3)
                    .add("sub_str",
                         "c[0][${_i}] = vfmaq_laneq_f32(c[0][${_i}], "
                         "weight[0][weight_idx],  src[(${_i} * ${stride} + "
                         "src_idx) / 4], (${_i} * ${stride} + src_idx) % 4);")
                    .render(temp_str);
    std::string expected_str =
            "c[0][0] = vfmaq_laneq_f32(c[0][0], weight[0][weight_idx], "
            " src[(0 * 1 + src_idx) / 4], (0 * 1 + src_idx) % 4);c[0][1] = "
            "vfmaq_laneq_f32(c[0][1], weight[0][weight_idx],  src[(1 * 1 + "
            "src_idx) / 4], (1 * 1 + src_idx) % 4);c[0][2] = "
            "vfmaq_laneq_f32(c[0][2], weight[0][weight_idx],  src[(2 * 1 + "
            "src_idx) / 4], (2 * 1 + src_idx) % 4);";
    EXPECT_TRUE(generted_str == expected_str);
    std::string temp_str2 = "${_unroll(3, sub_str, stride:1)}";
    auto generted_str2 =
            StringTemplate::StringTemplateArgs()
                    .add("sub_str",
                         "c[0][${_i}] = vfmaq_laneq_f32(c[0][${_i}], "
                         "weight[0][weight_idx],  src[(${_i} * ${stride} + "
                         "src_idx) / 4], (${_i} * ${stride} + src_idx) % 4);")
                    .render(temp_str2);
    EXPECT_TRUE(generted_str2 == expected_str);
}

#ifdef __x86_64__
//! TODO: can't test expect death in arm
TEST(TOOLS, ABORT) {
    EXPECT_DEATH(CC_ABORT << "it is about to abort\n", "");
    EXPECT_DEATH(CC_ASSERT(0) << "it is about to abort\n", "");
}
#endif