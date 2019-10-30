add_library(cudaCommon "")
target_sources(cudaCommon PRIVATE
  my_cuda_utils.cu
  OpenMPTimer.cpp
  SimpleTimer.cpp)

target_include_directories(cudaCommon
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_compile_options(cudaCommon PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(cudaCommon PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)

add_library(cuda::common ALIAS cudaCommon)
