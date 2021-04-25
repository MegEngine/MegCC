#include <math.h>
#include <string.h>
#include "stdio.h"
#include "stdlib.h"

#define EXAMPLE_ASSERT(exp_, ...) \
    if (!(exp_)) {                \
        printf(""__VA_ARGS__);    \
        __builtin_trap();         \
    }

/// swap left and right
static void swap(unsigned char* left, unsigned char* right) {
    unsigned char temp = *left;
    *left = *right;
    *right = temp;
}

/// get a new sedd from original seed
static unsigned int get_seed(unsigned int* m_seed) {
    *m_seed ^= *m_seed << 13;
    *m_seed ^= *m_seed >> 17;
    return *m_seed ^= *m_seed << 5;
}

/// water the given seed to get new data
static void water(unsigned int m_seed, unsigned char* str, size_t n) {
    unsigned char s[0x100];
    m_seed ^= n;
    for (int i = 0; i <= 0xFF; ++i)
        s[i] = i;
    for (int i = 0, j = 0; i <= 0xFF; ++i) {
        j = (j + s[i] + (get_seed(&m_seed) >> ((i & 3) << 3))) & 0xFF;
        swap(&s[i], &s[j]);
    }
    int i = 0, j = 0;
    for (int k = 0, nr_drop = (get_seed(&m_seed) & 4095) + 800; k < nr_drop;
         ++k) {
        i = (i + 1) & 0xFF;
        j = (j + s[i]) & 0xFF;
        swap(&s[i], &s[j]);
    }
    for (size_t k = 0; k < n; ++k) {
        i = (i + 1) & 0xFF;
        j = (j + s[i]) & 0xFF;
        swap(&s[i], &s[j]);
        str[k] ^= s[(s[i] + s[j]) & 0xFF];
    }
}

static int piecewise_normalize(float score, float left, float right,
                               float threshold, int new_left, int new_right,
                               int new_threshold) {
    if (!(score >= left && score <= right)) {
        return new_left;
    }
    int ret;
    if (score < threshold) {
        ret = ((score - left) / (threshold - left) *
                       (new_threshold - new_left) +
               new_left);
        if (ret < new_left)
            ret = new_left;
        if (ret >= new_threshold)
            ret = new_threshold - 1;
    } else {
        ret = ((score - threshold) / (right - threshold) *
                       (new_right - new_threshold) +
               new_threshold);
        if (ret <= new_threshold)
            ret = new_threshold + 1;
        if (ret > new_right)
            ret = new_right;
    }
    return ret;
}

/// the original seed
static unsigned int seed = 0x0F3A286E;
static void* read_file(const char* file_name) {
    FILE* fin = fopen(file_name, "rb");
    if (!fin) {
        fprintf(stderr, "Open file error!!\n");
        return NULL;
    }
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    void* ptr = malloc(size);
    size_t read_bytes = fread(ptr, 1, size, fin);
    fclose(fin);
    EXAMPLE_ASSERT(read_bytes == 4096);
    return ptr;
}
//! two bin file which has 1024 float, compute similarity score
int main(int argn, void** args) {
    float alpha = 2.297190;
    float beta = -2.659770;
    float threshold = 55.3;
    int feature_size = 4096;
    float* buffer_a = read_file((char*)(args[1]));
    float* buffer_b = read_file((char*)(args[2]));

    water(seed, (unsigned char*)buffer_a, feature_size);
    water(seed, (unsigned char*)buffer_b, feature_size);

    water(seed, (unsigned char*)buffer_a, feature_size);
    water(seed, (unsigned char*)buffer_b, feature_size);

    float score = 1e38f;
    size_t feature_size_in_float = feature_size / sizeof(float);
    /**
     * whole buffer is a complete feature without being seperate into two
     * segments
     */
    size_t start = 0;
    size_t end = feature_size_in_float;
    float distance = 0;
    for (size_t i = start; i < end; i++) {
        distance += (buffer_a[i] - buffer_b[i]) * (buffer_a[i] - buffer_b[i]);
    }
    float score_full = (float)(100.0f / (1.0 + exp(alpha * distance + beta)));

    score = piecewise_normalize(score_full, 0.f, 100.f, threshold, 0, 100, 80);

    printf("---->>>full_score=%f, score=%f, distance %f\n", score_full, score,
           distance);
    return 0;
}
