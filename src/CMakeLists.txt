add_subdirectory(common)

add_executable(saxpy_cuda "")
target_sources(saxpy_cuda PRIVATE saxpy.cu)

# the following line shouldn't be necessary as target_link_libraries
# with OpenMP::OpenMP_CXX should pass the correct flags for OpenMP
# but apparently, this is KO when using nvcc :(
# You can remove that if OpenMP is not required
# checked using cmake 3.14.3 (October 2019)
target_compile_options(saxpy_cuda PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

target_link_libraries(saxpy_cuda PUBLIC CUDA::cublas)
target_link_libraries(saxpy_cuda PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(saxpy_cuda PUBLIC cuda::common)
