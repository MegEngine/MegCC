macro(gen_lite_schema)
  set(FLATBUFFERS_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(CMAKE_CXX_FLAGS_ "${CMAKE_CXX_FLAGS}")
  check_cxx_compiler_flag("-Wno-suggest-override" CHECK_NO-SUGGEST-OVERRIDE)
  if(CHECK_NO-SUGGEST-OVERRIDE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
  endif()
  add_subdirectory(
    ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/flatbuffers
    ${CMAKE_BINARY_DIR}/third_party/MegEngine/third_party/flatbuffers)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_}")

  set(LITE_SCHEMA_DIR ${PROJECT_SOURCE_DIR}/../runtime/schema)
  file(GLOB_RECURSE LITE_SCHEMA_FILES ${LITE_SCHEMA_DIR}/*.fbs)

  set(LITE_SCHEMA_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/schema/include)

  build_flatbuffers(
    "${LITE_SCHEMA_FILES}"
    "${LITE_SCHEMA_DIR}"
    lite_runtime_schema_fbs
    ""
    "${LITE_SCHEMA_GEN_DIR}/schema/"
    ""
    "")
endmacro(gen_lite_schema)
