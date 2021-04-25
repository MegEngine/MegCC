configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MEGCC_TEST_DEPENDS FileCheck count not megcc-opt)

add_lit_testsuite(check-megcc "Running the megcc regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${MEGCC_TEST_DEPENDS})

add_lit_testsuites(MEGCC ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MEGCC_TEST_DEPENDS})

add_dependencies(megcc-test check-megcc)
