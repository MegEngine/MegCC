#include <sys/time.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "lite-c/common_enum_c.h"
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "lite-c/tensor_c.h"

#define LITE_CAPI_CHECK(error_, msg_) \
    if (error_) {                     \
        printf(msg_);                 \
        LITE_destroy_network(model);  \
        __builtin_trap();             \
    }

#define NMS_THRESH       0.65
#define BBOX_CONF_THRESH 0.3

constexpr int INPUT_W = 640;
constexpr int INPUT_H = 640;

const float color_list[80][3] = {
        {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125},
        {0.494, 0.184, 0.556}, {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933},
        {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300}, {0.600, 0.600, 0.600},
        {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
        {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000},
        {0.333, 0.333, 0.000}, {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000},
        {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000}, {0.667, 1.000, 0.000},
        {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
        {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500},
        {0.333, 0.000, 0.500}, {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500},
        {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500}, {0.667, 0.333, 0.500},
        {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
        {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500},
        {0.000, 0.333, 1.000}, {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000},
        {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000}, {0.333, 0.667, 1.000},
        {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
        {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000},
        {1.000, 0.333, 1.000}, {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000},
        {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000}, {0.833, 0.000, 0.000},
        {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
        {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000},
        {0.000, 1.000, 0.000}, {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333},
        {0.000, 0.000, 0.500}, {0.000, 0.000, 0.667}, {0.000, 0.000, 0.833},
        {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
        {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571},
        {0.714, 0.714, 0.714}, {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741},
        {0.314, 0.717, 0.741}, {0.50, 0.5, 0}};

static const char* class_names[] = {"person",        "bicycle",      "car",
                                    "motorcycle",    "airplane",     "bus",
                                    "train",         "truck",        "boat",
                                    "traffic light", "fire hydrant", "stop sign",
                                    "parking meter", "bench",        "bird",
                                    "cat",           "dog",          "horse",
                                    "sheep",         "cow",          "elephant",
                                    "bear",          "zebra",        "giraffe",
                                    "backpack",      "umbrella",     "handbag",
                                    "tie",           "suitcase",     "frisbee",
                                    "skis",          "snowboard",    "sports ball",
                                    "kite",          "baseball bat", "baseball glove",
                                    "skateboard",    "surfboard",    "tennis racket",
                                    "bottle",        "wine glass",   "cup",
                                    "fork",          "knife",        "spoon",
                                    "bowl",          "banana",       "apple",
                                    "sandwich",      "orange",       "broccoli",
                                    "carrot",        "hot dog",      "pizza",
                                    "donut",         "cake",         "chair",
                                    "couch",         "potted plant", "bed",
                                    "dining table",  "toilet",       "tv",
                                    "laptop",        "mouse",        "remote",
                                    "keyboard",      "cell phone",   "microwave",
                                    "oven",          "toaster",      "sink",
                                    "refrigerator",  "book",         "clock",
                                    "vase",          "scissors",     "teddy bear",
                                    "hair drier",    "toothbrush"};

struct OutputInfo {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct BaseAnchor {
    int grid0;
    int grid1;
    int stride;
};

static void preprocess(cv::Mat& img, float* data_ptr) {
    float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat resize_img(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(resize_img(cv::Rect(0, 0, re.cols, re.rows)));

    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    int channels = 3;
    int img_h = resize_img.rows;
    int img_w = resize_img.cols;
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < img_h; h++) {
            for (size_t w = 0; w < img_w; w++) {
                data_ptr[c * img_w * img_h + h * img_w + w] =
                        (((float)resize_img.at<cv::Vec3b>(h, w)[c]) / 255.0f -
                         mean[c]) /
                        std[c];
            }
        }
    }
}

static inline float intersection_area(const OutputInfo& a, const OutputInfo& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(
        std::vector<OutputInfo>& faceoutput_infos, int left, int right) {
    int i = left;
    int j = right;
    float p = faceoutput_infos[(left + right) / 2].prob;

    while (i <= j) {
        while (faceoutput_infos[i].prob > p)
            i++;

        while (faceoutput_infos[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceoutput_infos[i], faceoutput_infos[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceoutput_infos, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceoutput_infos, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<OutputInfo>& output_infos) {
    if (output_infos.empty())
        return;

    qsort_descent_inplace(output_infos, 0, output_infos.size() - 1);
}

static void nms_sorted_bboxes(
        const std::vector<OutputInfo>& faceoutput_infos,
        std::vector<int>& picked_output, float nms_threshold) {
    picked_output.clear();

    const int n = faceoutput_infos.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceoutput_infos[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const OutputInfo& a = faceoutput_infos[i];

        int keep = 1;
        for (int j = 0; j < (int)picked_output.size(); j++) {
            const OutputInfo& b = faceoutput_infos[picked_output[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked_output[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked_output.push_back(i);
    }
}

static void postpreprocess(
        const float* output_ptr, std::vector<OutputInfo>& output_infos, float scale,
        const int img_w, const int img_h) {
    std::vector<OutputInfo> valid_output_info;
    std::vector<BaseAnchor> base_anchors;

    // get base anchor
    for (auto stride : {8, 16, 32}) {
        int num_anchor = INPUT_W / stride;
        for (int g1 = 0; g1 < num_anchor; g1++) {
            for (int g0 = 0; g0 < num_anchor; g0++) {
                base_anchors.push_back((BaseAnchor){g0, g1, stride});
            }
        }
    }
    // get valid output information
    const int num_class = 80;
    const int num_anchors = base_anchors.size();
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int grid0 = base_anchors[anchor_idx].grid0;
        const int grid1 = base_anchors[anchor_idx].grid1;
        const int stride = base_anchors[anchor_idx].stride;

        const int basic_pos = anchor_idx * 85;

        float x_center = (output_ptr[basic_pos + 0] + grid0) * stride;
        float y_center = (output_ptr[basic_pos + 1] + grid1) * stride;
        float w = exp(output_ptr[basic_pos + 2]) * stride;
        float h = exp(output_ptr[basic_pos + 3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = output_ptr[basic_pos + 4];
        for (int class_idx = 0; class_idx < num_class; class_idx++) {
            float box_cls_score = output_ptr[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > BBOX_CONF_THRESH) {
                OutputInfo obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                valid_output_info.push_back(obj);
            }

        }  // class loop

    }  // point anchor loop

    qsort_descent_inplace(valid_output_info);

    std::vector<int> picked_output;
    nms_sorted_bboxes(valid_output_info, picked_output, NMS_THRESH);
    int count = picked_output.size();
    output_infos.resize(count);

    for (int i = 0; i < count; i++) {
        output_infos[i] = valid_output_info[picked_output[i]];

        // adjust offset to original unpadded
        float x0 = (output_infos[i].rect.x) / scale;
        float y0 = (output_infos[i].rect.y) / scale;
        float x1 = (output_infos[i].rect.x + output_infos[i].rect.width) / scale;
        float y1 = (output_infos[i].rect.y + output_infos[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        output_infos[i].rect.x = x0;
        output_infos[i].rect.y = y0;
        output_infos[i].rect.width = x1 - x0;
        output_infos[i].rect.height = y1 - y0;
    }
}

static void draw_output_infos(
        const cv::Mat& bgr, const std::vector<OutputInfo>& output_infos,
        std::string& output_name) {
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < output_infos.size(); i++) {
        const OutputInfo& obj = output_infos[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(
                color_list[obj.label][0], color_list[obj.label][1],
                color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        } else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size =
                cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        if (y > image.rows)
            y = image.rows;
        cv::rectangle(
                image,
                cv::Rect(
                        cv::Point(x, y),
                        cv::Size(label_size.width, label_size.height + baseLine)),
                txt_bk_color, -1);

        cv::putText(
                image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite(output_name.c_str(), image);
    std::cout << "save output to " << output_name << std::endl;
}

std::vector<float> run_model(LiteNetwork model) {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    LITE_CAPI_CHECK(LITE_forward(model), "run model failed\n");
    LITE_CAPI_CHECK(LITE_wait(model), "wait model failed\n");
    gettimeofday(&end, NULL);
    unsigned long diff =
            1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("every iter use time: %fms\n", ((float)diff) / 1000);
    LiteTensor output;
    size_t nr_output = 0;
    LITE_CAPI_CHECK(
            LITE_get_all_output_name(model, &nr_output, NULL),
            "get output number failed\n");
    char* output_name_ptr[nr_output];
    LITE_CAPI_CHECK(
            LITE_get_all_output_name(model, NULL, (const char**)&output_name_ptr),
            "get output name failed\n");
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(model, output_name_ptr[0], LITE_OUTPUT, &output),
            "get output tensor failed\n");
    size_t length = 0;
    LITE_CAPI_CHECK(
            LITE_get_tensor_total_size_in_byte(output, &length),
            "get output tensor size failed\n");
    float* output_ptr = NULL;
    LITE_CAPI_CHECK(
            LITE_get_tensor_memory(output, (void**)&output_ptr),
            "get output tensor memory failed\n");

    std::vector<float> output_vec(output_ptr, output_ptr + length / sizeof(float));

    return output_vec;
}

std::vector<float> inference(const char* model_path, std::vector<float>& input_data) {
    LITE_set_log_level(INFO);
    LiteNetwork model;
    LiteTensor input;
    size_t input_length;
    //! make network
    LITE_CAPI_CHECK(
            LITE_make_network(&model, *default_config(), *default_network_io()),
            "create model error. \n");
    LITE_CAPI_CHECK(
            LITE_load_model_from_path(model, model_path), "load model error. \n");
    //! set input tensor
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(model, "data", LITE_INPUT, &input),
            "get input tensor failed\n");
    LITE_CAPI_CHECK(
            LITE_get_tensor_total_size_in_byte(input, &input_length),
            "get input tensor size failed\n");

    LITE_CAPI_CHECK(
            LITE_reset_tensor_memory(input, input_data.data(), input_length),
            "set input ptr failed\n");

    LITE_CAPI_CHECK(LITE_destroy_tensor(input), "destory input tensor");

    std::vector<float> output = run_model(model);
    LITE_CAPI_CHECK(LITE_destroy_network(model), "delete model failed\n");
    return output;
}

std::string keys =
        "{ @model          | | the inference model path}"
        "{ input           | | the input image }"
        "{ output          | | the output image }";

static void help(char** argv) {
    std::cout << "\nYOLOX inference example \n"
                 "Using OpenCV version: \n"
              << CV_VERSION
              << "\n"
                 "Usage:\n"
              << argv[0] << " <model_name>"
              << " --input=<input_image>"
              << " --input=<output_image>\n"
              << std::endl;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, keys);
    if (argc < 3) {
        help(argv);
        return -1;
    }
    const std::string model_path = parser.get<cv::String>("@model");
    std::string image_filename = cv::samples::findFile(parser.get<cv::String>("input"));
    std::string outout_filename = parser.get<cv::String>("output");

    cv::Mat img = cv::imread(image_filename);
    if (img.empty()) {
        std::cout << "Couldn't open the image " << image_filename
                  << ". Usage: inpaint <image_name>\n"
                  << std::endl;
        return 0;
    }
    std::vector<float> input(3 * 640 * 640, 0);
    preprocess(img, input.data());
    std::vector<float> output_data = inference(model_path.c_str(), input);
    float* predict_ptr = output_data.data();
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    std::vector<OutputInfo> output_infos;

    postpreprocess(predict_ptr, output_infos, scale, img_w, img_h);
    draw_output_infos(img, output_infos, outout_filename);
    return 0;
}
