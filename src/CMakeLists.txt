add_executable(saxpy_cuda "")
target_sources(saxpy_cuda PRIVATE 
  saxpy.cu
  my_cuda_utils.cu
  SimpleTimer.cpp)
target_link_libraries(saxpy_cuda PRIVATE CUDA::cublas)
