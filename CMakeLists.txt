cmake_minimum_required(VERSION 3.24)

project(torch-mini-cpp
  VERSION 0.1.0
  LANGUAGES CXX)

# Options
option(TORCH_MINI_ENABLE_TESTING "Build tests" ON)
option(TORCH_MINI_BUILD_APPS "Build example/CLI apps" OFF)
option(TORCH_MINI_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(TORCH_MINI_ENABLE_SANITIZERS "Enable sanitizers on Debug builds" OFF)
option(TORCH_MINI_ENABLE_IPO "Enable LTO/IPO on Release builds (if supported)" ON)

# Helpful defaults
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Modules with helper functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

# Library code
add_subdirectory(src)

# Testing
if(TORCH_MINI_ENABLE_TESTING)
  include(CTest)  # adds BUILD_TESTING option and CTest integration
  enable_testing()

  include(FetchContent)
  # GoogleTest (modern: via FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    # Or change to a newer release if you like
  )
  # Recommended: use the same CRT on MSVC
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests)
endif()

# (Optional) apps later
if(TORCH_MINI_BUILD_APPS)
  # add_subdirectory(apps)  # keep empty for M0
endif()