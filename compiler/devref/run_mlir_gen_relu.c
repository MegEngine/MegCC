#include <stdio.h>
#include <stdlib.h>

typedef struct MemRef_descriptor_* MemRef_descriptor;
typedef struct MemRef_descriptor_ {
    float* allocated;
    float* aligned;
    size_t offset;
    size_t sizes[2];
    size_t strides[2];
} Memref;
#define FUNC_NAME _mlir_ciface_my_codegen_elem

extern void FUNC_NAME(Memref* a, Memref* c);

#define ALEN 10
int main() {
    float a[ALEN];
    float c[ALEN];
    for (int i = 0; i < ALEN; ++i) {
        a[i] = -5.f + i;
        c[i] = 0;
    }
    Memref ref_a, ref_c;
    ref_a.aligned = &a[0];
    ref_a.allocated = &a[0];
    ref_a.offset = 0;
    ref_a.sizes[0] = 1;
    ref_a.sizes[1] = ALEN;
    ref_a.strides[0] = ALEN;
    ref_a.strides[0] = 1;
    ref_c.aligned = &c[0];
    ref_c.allocated = &c[0];
    ref_c.offset = 0;
    ref_c.sizes[0] = 1;
    ref_c.sizes[1] = ALEN;
    ref_c.strides[0] = ALEN;
    ref_c.strides[1] = 1;
    FUNC_NAME(&ref_a, &ref_c);
    for (int i = 0; i < ALEN; ++i) {
        printf("%f, ", a[i]);
    }
    return 0;
}