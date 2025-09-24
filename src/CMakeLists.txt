add_library(torch_mini_core STATIC)

# Public headers live under include/
target_sources(torch_mini_core
  PRIVATE
    eltwise.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
      ${CMAKE_SOURCE_DIR}/include/eltwise.hpp
)

# C++ standard per-target (modern style)
target_compile_features(torch_mini_core PUBLIC cxx_std_23)

# Include dirs for consumers (our tests/apps)
target_include_directories(torch_mini_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Warnings / Sanitizers
torch_mini_enable_warnings(torch_mini_core ${TORCH_MINI_WARNINGS_AS_ERRORS})
torch_mini_enable_sanitizers(torch_mini_core)

# IPO/LTO for Release if supported
if(TORCH_MINI_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
  if(ipo_supported)
    set_property(TARGET torch_mini_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endif()

# Nice alias for consumers
add_library(torch_mini::core ALIAS torch_mini_core)
