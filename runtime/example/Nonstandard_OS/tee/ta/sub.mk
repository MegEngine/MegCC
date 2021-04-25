global-incdirs-y += include
global-incdirs-y += tinynn_sdk_install/include
global-incdirs-y += ../common_api
srcs-y += megcc_inference_ta.c
srcs-y += api.c
srcs-y += ../common_api/api.c
libnames += TinyNN
libdirs += tinynn_sdk_install/lib
libdeps += tinynn_sdk_install/lib/libTinyNN.a
libnames += m
libdirs += none_libm
libdeps += none_libm/libm.a

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
