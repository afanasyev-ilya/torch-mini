add_executable(test_eltwise test_eltwise.cpp)
target_compile_features(test_eltwise PRIVATE cxx_std_23)
target_link_libraries(test_eltwise PRIVATE
  torchmini::core
  GTest::gtest_main
)

# Apply warnings/sanitizers to tests too
#torch_mini_enable_warnings(test_eltwise ${MINITORCH_WARNINGS_AS_ERRORS})
#torch_mini_enable_sanitizers(test_eltwise)

include(GoogleTest)
gtest_discover_tests(test_eltwise
  DISCOVERY_MODE PRE_TEST
)
