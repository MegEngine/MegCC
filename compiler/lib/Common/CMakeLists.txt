file(GLOB_RECURSE SOURCES_FILE *.cpp *.h)
add_library(Common STATIC ${SOURCES_FILE})
target_include_directories(Common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)
target_compile_options(Common PRIVATE -Werror)
