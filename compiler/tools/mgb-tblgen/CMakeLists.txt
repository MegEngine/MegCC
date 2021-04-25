set(LLVM_LINK_COMPONENTS Support TableGen)
file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_tablegen(mgb-tblgen MGB ${SRCS})
target_link_libraries(mgb-tblgen PRIVATE MLIRTableGen)
