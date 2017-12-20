include config.h

all: target

target: get_gpu_info stencil_comp \
isofd3d_gpu_kernel.o \
fdtd3d_gpu \
cuda_array_test

get_gpu_info: get_gpu_info.o
	$(NVCC) $(LDFLAGS) $(CUDA_FLAGS) -o $@ $+ $(LIBRARIES)

get_gpu_info.o: get_gpu_info.cpp
	$(NVCC) $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<

isofd3d_gpu_kernel.o: isofd3d_gpu_kernel.cu
	$(NVCC) $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<

iso_fd_comp.o: iso_fd_comp.cu
	$(NVCC) $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<

cudaArray.o: cudaArray.cu
	$(NVCC) $(INCLUDES) $(CUDA_FLAGS) -o $@ -c $<

stencil_comp: sc3d.o isofd3d_gpu_kernel.o
	$(CXX) -o $@ -fopenmp $+ $(LDFLAGS) $(LIBRARIES)

fdtd3d_gpu: iso_fd_comp.o
	$(NVCC) $(LDFLAGS) $(CUDA_FLAGS) -o $@ $+ $(LIBRARIES)

cuda_array_test: cudaArray.o
	$(NVCC) $(LDFLAGS) $(CUDA_FLAGS) -o $@ $+ $(LIBRARIES)

.cpp.o:
	$(CXX) $(INCLUDES) -c $(CXXFLAGS) $<

.c.o:
	$(CC) $(INCLUDES) -c $(CXXFLAGS) $<


.PHONY: clean
clean:
	-rm -f *.o
	-rm -f get_gpu_info
	-rm -f stencil_comp
	-rm -f fdtd3d_gpu
