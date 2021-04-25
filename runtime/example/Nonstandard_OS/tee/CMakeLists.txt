project(optee_example_megcc_inference C)

set(SRC host/main.c common_api/api.c)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ta/include
  PRIVATE common_api
  PRIVATE ta/tinynn_sdk_install/include
  PRIVATE include)

set(tinynn_static_lib ${PROJECT_SOURCE_DIR}/ta/tinynn_sdk_install/lib/libTinyNN.a)
target_link_libraries(${PROJECT_NAME} PRIVATE teec ${tinynn_static_lib})

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
