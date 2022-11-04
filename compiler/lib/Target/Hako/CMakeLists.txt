file(GLOB_RECURSE SOURCES_FILE *.cpp *.h)
add_library(HakoParse STATIC ${SOURCES_FILE})
target_include_directories(HakoParse PRIVATE ${PROJECT_SOURCE_DIR}/include/ decryption/)
target_compile_options(HakoParse PRIVATE -Werror)
