#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
//#include "cuPrintf.cu"
#include <limits.h>

#define CUDA_VERSION 4

#define GNU_C_COMPILER
#if defined(GNU_C_COMPILER)
extern "C" {
#include "cblas.h"
#include "lapacke.h"
#include "lapacke_mangling.h"
}
#elif defined(INTEL_C_COMPILER)
#include "fftw3.h"
#include "mkl.h"
#endif

using std::cout;
using std::complex;

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#ifndef EXIT_USAGE
#define EXIT_USAGE -1
#endif
#define nullptr NULL

#define BLOCK_SIZE 2048
#define THREAD_SIZE 256
#define DEBUG
#define VERBOSITY

#define BLOCK_X 128
#define BLOCK_Y 128
#define BLOCK_Z 128

#define THREAD_X 32
#define THREAD_Y 8
#define THREAD_Z 2

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

inline void __safeCall(cudaError err, const char * file, const int line)
{
	if(cudaSuccess != err) {
		fprintf(stderr, "ERROR: safeCall() Runtime API error in file <%s>, line %i : %s.\n", file , line, cudaGetErrorString(err));
		exit(-1);
	}
}

class TimerGPU {
public:
	cudaEvent_t start, stop;
	cudaStream_t stream;
	TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}
	~TimerGPU() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	float read() {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
};

class TimerCPU {
	static const int bits = 10;
public:
	long long beg_clock;
	float freq;
	TimerCPU(float freq_) : freq(freq_) { 
		beg_clock = getTSC(bits);
	}
	long long getTSC(int bits) {
#ifdef WIN32
		return __rdtsc();
#else
		unsigned int low, high;
		__asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
		return ((long long)high<<(32 - bits)) | ((long long)low >> bits);
#endif
	}
	float read() {
		long long end_clock = getTSC(bits);
		long long Kcycles = end_clock - beg_clock;
		float time = (float)(1 << bits) * Kcycles / freq / 1e3f;
		return time;
	}
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

int iDivUp(int a, int b) { return (a % b == 0) ? (a / b) : (a / b + 1); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b == 0) ? a : (a - a % b + b); }
int iAlignDown(int a, int b) { return a - a % b; }

template<typename T> class CudaArray;
template<typename T> class ExpressionUnary;
template<typename Tleft, typename Tright> class ExpressionBinary;

#define CUDA_ARRAY_SIZE_TYPE unsigned long long
template<size_t BSIZE, size_t TSIZE, typename T>
__global__ static void cuda_array_plus(T * od, const T * left, const T * right, const CUDA_ARRAY_SIZE_TYPE n) {
	const CUDA_ARRAY_SIZE_TYPE tx = threadIdx.x;
	const CUDA_ARRAY_SIZE_TYPE gx = (blockIdx.y * gridDim.x + blockIdx.x) * BSIZE;

	od += gx;
	left += gx;
	right += gx;
	CUDA_ARRAY_SIZE_TYPE i;
	#pragma unroll
	for(i = tx; i < BSIZE; i += TSIZE) {
		if(gx + i < n) od[i] = left[i] + right[i];
	}
}

template<size_t BSIZE, size_t TSIZE, typename T>
__global__ static void cuda_array_sub(T * od, const T * left, const T * right, const CUDA_ARRAY_SIZE_TYPE n) {
	const CUDA_ARRAY_SIZE_TYPE tx = threadIdx.x;
	const CUDA_ARRAY_SIZE_TYPE gx = (blockIdx.y * gridDim.x + blockIdx.x) * BSIZE;

	od += gx;
	left += gx;
	right += gx;
	CUDA_ARRAY_SIZE_TYPE i;
	#pragma unroll
	for(i = tx; i < BSIZE; i += TSIZE) {
		if(gx + i < n) od[i] = left[i] - right[i];
	}
}

template<size_t BSIZE, size_t TSIZE, typename T>
__global__ static void cuda_array_mul(T * od, const T * left, const T * right, const CUDA_ARRAY_SIZE_TYPE n) {
	const CUDA_ARRAY_SIZE_TYPE tx = threadIdx.x;
	const CUDA_ARRAY_SIZE_TYPE gx = (blockIdx.y * gridDim.x + blockIdx.x) * BSIZE;

	od += gx;
	left += gx;
	right += gx;
	CUDA_ARRAY_SIZE_TYPE i;
	#pragma unroll
	for(i = tx; i < BSIZE; i += TSIZE) {
		if(gx + i < n) od[i] = left[i] * right[i];
	}
}

template<size_t BSIZE, size_t TSIZE, typename T>
__global__ static void cuda_array_div(T * od, const T * left, const T * right, const CUDA_ARRAY_SIZE_TYPE n) {
	const CUDA_ARRAY_SIZE_TYPE tx = threadIdx.x;
	const CUDA_ARRAY_SIZE_TYPE gx = (blockIdx.y * gridDim.x + blockIdx.x) * BSIZE;

	od += gx;
	left += gx;
	right += gx;
	CUDA_ARRAY_SIZE_TYPE i;
	#pragma unroll
	for(i = tx; i < BSIZE; i += TSIZE) {
		if(gx + i < n) od[i] = left[i] / right[i];
	}
}

template<size_t BSIZE, size_t TSIZE>
__global__ static void cuda_array_cmulbyreal(Complex * od, Complex * left, const float * right, const CUDA_ARRAY_SIZE_TYPE n) {
	const CUDA_ARRAY_SIZE_TYPE tx = threadIdx.x;
	const CUDA_ARRAY_SIZE_TYPE gx = (blockIdx.y * gridDim.x + blockIdx.x) * BSIZE;

	od += gx;
	left += gx;
	right += gx;
	CUDA_ARRAY_SIZE_TYPE i;
	#pragma unroll
	for(i = tx; i < BSIZE; i += TSIZE) {
		if(gx + i < n) od[i] = ComplexScale(left[i], right[i]);
	}
}

template<size_t BSIZE, size_t TSIZE, typename T>
__global__ static void cuda_array_fill(T * od, const CUDA_ARRAY_SIZE_TYPE n, T val) {
	const CUDA_ARRAY_SIZE_TYPE tx = threadIdx.x;
	const CUDA_ARRAY_SIZE_TYPE gx = (blockIdx.y * gridDim.x + blockIdx.x) * BSIZE;
	
	od += gx;
	CUDA_ARRAY_SIZE_TYPE i;
	#pragma unroll
	for(i = tx; i < BSIZE; i += TSIZE) {
		if(gx + i < n) od[i] = val;
	}
}

enum BinaryOperationType { PLUS, SUB, MUL, DIV, CMULBYREAL };
template<typename Tleft, typename Tright>
class ExpressionBinary {
public:
	ExpressionBinary() : left(nullptr), right(nullptr) { 
	}
	ExpressionBinary(CudaArray<Tleft> * __left, CudaArray<Tright> * __right, BinaryOperationType __type) : left(__left), right(__right), type(__type) { 
	}
	ExpressionBinary(const ExpressionBinary& exp) {
		left = exp.left;
		right = exp.right;
		type = exp.type;
	}
	ExpressionBinary& operator=(const ExpressionBinary& exp) {
		left = exp.left;
		right = exp.right;
		type = exp.type;
		return *this;
	}
	~ExpressionBinary() { left = nullptr; right = nullptr;};
public:
	CudaArray<Tleft> * left;
	CudaArray<Tright> * right;
	BinaryOperationType type;
};

template<typename T>
ExpressionBinary<T, T> operator+(CudaArray<T>& left, CudaArray<T>& right) {
	return ExpressionBinary<T, T>(&left, &right, PLUS);
}

template<typename T>
ExpressionBinary<T, T> operator-(CudaArray<T>& left, CudaArray<T>& right) {
	return ExpressionBinary<T, T>(&left, &right, SUB);
}

template<typename T>
ExpressionBinary<T, T> operator*(CudaArray<T>& left, CudaArray<T>& right) {
	return ExpressionBinary<T, T>(&left, &right, MUL);
}

template<typename T>
ExpressionBinary<T, T> operator/(CudaArray<T>& left, CudaArray<T>& right) {
	return ExpressionBinary<T, T>(&left, &right, DIV);
}

ExpressionBinary<Complex, float> operator*(CudaArray<Complex>& left, CudaArray<float>& right) {
	return ExpressionBinary<Complex, float>(&left, &right, CMULBYREAL);
}

template<typename T>
class CudaArray {
public:
	CudaArray() : n(0), data_ptr(nullptr), ref_count(nullptr) { 
#ifdef DEBUG
		fprintf(stderr, "INFO: default constructor of CudaArray<T> is called\n");
#endif
	}
	CudaArray(const CUDA_ARRAY_SIZE_TYPE __n) : n(__n) {
#ifdef DEBUG
		fprintf(stderr, "INFO: constructor CudaArray(const CUDA_ARRAY_SIZE_TYPE n) is called\n");
#endif
		safeCall(cudaMalloc((void**)&data_ptr, (long long int)n * sizeof(T)));
		if(data_ptr == nullptr) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", (long long int)n * sizeof(T), __FILE__, __LINE__);	
		}
		ref_count = new int(1);
	}
	CudaArray(const CudaArray<T>& arr) : n(arr.n) {
#ifdef DEBUG
		fprintf(stderr, "INFO: copy constructor of CudaArray<T> is called\n");
#endif
		data_ptr = arr.data_ptr;
		ref_count = arr.ref_count;
		if(ref_count != nullptr) *ref_count += 1;
	}
	CudaArray& operator=(const CudaArray<T>& arr) {
#ifdef DEBUG
		fprintf(stderr, "INFO: assignment operator of CudaArray<T> is called\n");
#endif
		n = arr.n;
		if(data_ptr != arr.data_ptr) {
			if(data_ptr != nullptr) {
				*ref_count -= 1;
				if(*ref_count == 0) {
					safeCall(cudaFree(data_ptr));
					data_ptr = nullptr;
					delete ref_count;
					ref_count = nullptr;
				}
			}
			data_ptr = arr.data_ptr;
			ref_count = arr.ref_count;
			if(ref_count != nullptr) *ref_count += 1;
		}
		return *this;
	}
#if CUDA_VERSION >= 7
	CudaArray(CudaArray<T>&& temp) : n(temp.n) {
#ifdef DEBUG
		fprintf(stderr, "INFO: move constructor of CudaArray<T> is called\n");
#endif
		data_ptr = temp.data_ptr;
		ref_count = temp.ref_count;
		temp.n = 0;
		if(temp.data_ptr != nullptr) {
			temp.data_ptr = nullptr;
			temp.ref_count = nullptr;
		}
	}
	CudaArray& operator=(CudaArray<T>&& temp) {
#ifdef DEBUG
		fprintf(stderr, "INFO: move assignment operator of CudaArray<T> is called\n");
#endif
		n = temp.n;
		if(data_ptr != temp.data_ptr) {
			if(data_ptr != nullptr) {
				*ref_count -= 1;
				if(*ref_count == 0) {
					safeCall(cudaFree(data_ptr));
					data_ptr = nullptr;
					delete ref_count;
					ref_count = nullptr;
				}
			}
			data_ptr = temp.data_ptr;
			ref_count = temp.ref_count;
			temp.n = 0;
			if(temp.data_ptr ! = nullptr) {
				temp.data_ptr = nullptr;
				temp.ref_count = nullptr;
			}
		} else {
			temp.n = 0;
		}
		return *this;		
	}
#endif
	~CudaArray() {
#ifdef DEBUG
		fprintf(stderr, "INFO: deconstructor of CudaArray<T> is called\n");
#endif
		if(ref_count != nullptr) {
			if(*ref_count > 1) *ref_count -= 1;
			else if(*ref_count == 1) {
				fprintf(stderr, "INFO: free memory of cuda array\n");
				delete ref_count;
				ref_count = nullptr;
				safeCall(cudaFree(data_ptr));
				data_ptr = nullptr;
			}
		}
	}
	CudaArray& operator=(const ExpressionBinary<T, T>& bin_op) {
#ifdef DEBUG
		fprintf(stderr, "INFO: assignment operator =(ExpressionBinary&) of CudaArray<T> is called\n");
#endif
		assert((*(bin_op.left)).size() == (*(bin_op.right)).size());
		const CUDA_ARRAY_SIZE_TYPE osize = (*(bin_op.left)).size();
		if(n != osize) {
			n = osize;
			if(data_ptr != nullptr) {
				*ref_count -= 1;
				if(ref_count == 0) {
					safeCall(cudaFree(data_ptr));
					data_ptr = nullptr;
					delete ref_count;
					ref_count = nullptr;	
				}
			}
			safeCall(cudaMalloc((void**)&data_ptr, (long long int)n * sizeof(T)));
			ref_count = new int(1);
			if(data_ptr == nullptr) {
				fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", (long long int)n * sizeof(T), __FILE__, __LINE__);
			}
		}
		int threads = THREAD_SIZE;
#define MAX_BLOCK_DIM 65535
		int k = iDivUp(n, MAX_BLOCK_DIM * BLOCK_SIZE);
		int blocks = iDivUp(n, k * BLOCK_SIZE);
		fprintf(stderr, "INFO: folds = %d, number of blocks = %d\n", k, blocks);
		TimerGPU timer(0);

#define CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(token, bsize) cuda_array_##token<bsize, THREAD_SIZE, T><<<blocks, threads>>>(data_ptr, (*(bin_op.left)).data_ptr, (*(bin_op.right)).data_ptr, n);
		if(bin_op.type == PLUS) {
			if(k == 1) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 1 * BLOCK_SIZE)
			else if(k == 2) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 2 * BLOCK_SIZE)
			else if(k == 3) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 3 * BLOCK_SIZE)
			else if(k == 4) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 4 * BLOCK_SIZE)
			else if(k == 5) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 5 * BLOCK_SIZE)
			else if(k == 6) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 6 * BLOCK_SIZE)
			else if(k == 7) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 7 * BLOCK_SIZE)
			else if(k == 8) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(plus, 8 * BLOCK_SIZE)
		}
		else if(bin_op.type == SUB) {
			if(k == 1) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 1 * BLOCK_SIZE)
			else if(k == 2) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 2 * BLOCK_SIZE)
			else if(k == 3) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 3 * BLOCK_SIZE)
			else if(k == 4) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 4 * BLOCK_SIZE)
			else if(k == 5) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 5 * BLOCK_SIZE)
			else if(k == 6) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 6 * BLOCK_SIZE)
			else if(k == 7) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 7 * BLOCK_SIZE)
			else if(k == 8) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(sub, 8 * BLOCK_SIZE)
		}
		else if(bin_op.type == MUL) {
			if(k == 1) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 1 * BLOCK_SIZE)
			else if(k == 2) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 2 * BLOCK_SIZE)
			else if(k == 3) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 3 * BLOCK_SIZE)
			else if(k == 4) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 4 * BLOCK_SIZE)
			else if(k == 5) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 5 * BLOCK_SIZE)
			else if(k == 6) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 6 * BLOCK_SIZE)
			else if(k == 7) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 7 * BLOCK_SIZE)
			else if(k == 8) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(mul, 8 * BLOCK_SIZE)
		}
		else if(bin_op.type == DIV) {
			if(k == 1) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 1 * BLOCK_SIZE)
			else if(k == 2) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 2 * BLOCK_SIZE)
			else if(k == 3) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 3 * BLOCK_SIZE)
			else if(k == 4) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 4 * BLOCK_SIZE)
			else if(k == 5) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 5 * BLOCK_SIZE)
			else if(k == 6) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 6 * BLOCK_SIZE)
			else if(k == 7) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 7 * BLOCK_SIZE)
			else if(k == 8) CUDA_ARRAY_ELEMENT_WISE_KERNEL_CALL(div, 8 * BLOCK_SIZE)
		}
		double op_time = timer.read();
#ifdef VERBOSITY
		fprintf(stderr, "INFO: binary operator gpu time: %.2f ms\n", op_time);
		fprintf(stderr, "INFO: binary operator arimetic performance: %.2f GFLOPS\n", n / (1e9 * op_time / 1e3));
		fprintf(stderr, "INFO: binary operator dram write performance: %.2f GB/s\n", 4 * n / (1e9 * op_time / 1e3));
		fprintf(stderr, "INFO: binary operator dram read performance: %.2f GB/s\n", 8 * n / (1e9 * op_time / 1e3));
#endif	
		return *this;	
	}

	CudaArray& operator=(const ExpressionBinary<Complex, float>& bin_op) {
		assert((*(bin_op.left)).size() == (*(bin_op.right)).size());
		const CUDA_ARRAY_SIZE_TYPE osize = (*(bin_op.left)).size();
		if(n != osize) {
			n = osize;
			if(data_ptr != nullptr) {
				*ref_count -= 1;
				if(ref_count == 0) {
					safeCall(cudaFree(data_ptr));
					data_ptr = nullptr;
					delete ref_count;
					ref_count = nullptr;	
				}
			}
			safeCall(cudaMalloc((void**)&data_ptr, (long long int)n * sizeof(T)));
			ref_count = new int(1);
			if(data_ptr == nullptr) {
				fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", (long long int)n * sizeof(T), __FILE__, __LINE__);
			}
		}
		int threads = THREAD_SIZE;
		int k = iDivUp(n, MAX_BLOCK_DIM * BLOCK_SIZE);
		int blocks = iDivUp(n, k * BLOCK_SIZE);

#define CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(bsize) cuda_array_cmulbyreal<bsize, THREAD_SIZE><<<blocks, threads>>>(data_ptr, (*(bin_op.left)).data_ptr, (*(bin_op.right)).data_ptr, n);
		assert(bin_op.type == CMULBYREAL);
		if(k == 1) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(1 * BLOCK_SIZE)
		else if(k == 2) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(2 * BLOCK_SIZE)
		else if(k == 3) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(3 * BLOCK_SIZE)
		else if(k == 4) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(4 * BLOCK_SIZE)
		else if(k == 5) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(5 * BLOCK_SIZE)
		else if(k == 6) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(6 * BLOCK_SIZE)
		else if(k == 7) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(7 * BLOCK_SIZE)
		else if(k == 8) CUDA_ARRAY_CMULBYREAL_KERNEL_CALL(8 * BLOCK_SIZE)
		
		return *this;	
	}
	void fill(T val) {
		int threads = THREAD_SIZE;
		int k = iDivUp(n, MAX_BLOCK_DIM * BLOCK_SIZE);
		int blocks = iDivUp(n, k * BLOCK_SIZE);
		fprintf(stderr, "INFO: folds = %d, number of blocks = %d\n", k, blocks);
		TimerGPU timer(0);
#define CUDA_ARRAY_FILL_KERNEL_CALL(bsize) cuda_array_fill<bsize, THREAD_SIZE, T><<<blocks, threads>>>(data_ptr, n, val);	
		if(k == 1) CUDA_ARRAY_FILL_KERNEL_CALL(1 * BLOCK_SIZE)
		else if(k == 2) CUDA_ARRAY_FILL_KERNEL_CALL(2 * BLOCK_SIZE)
		else if(k == 3) CUDA_ARRAY_FILL_KERNEL_CALL(3 * BLOCK_SIZE)
		else if(k == 4) CUDA_ARRAY_FILL_KERNEL_CALL(4 * BLOCK_SIZE)
		else if(k == 5) CUDA_ARRAY_FILL_KERNEL_CALL(5 * BLOCK_SIZE)
		else if(k == 6) CUDA_ARRAY_FILL_KERNEL_CALL(6 * BLOCK_SIZE)
		else if(k == 7) CUDA_ARRAY_FILL_KERNEL_CALL(7 * BLOCK_SIZE)
		else if(k == 8) CUDA_ARRAY_FILL_KERNEL_CALL(8 * BLOCK_SIZE)
		double op_time = timer.read();
#ifdef VERBOSITY
		fprintf(stderr, "INFO: fill with val %f gpu time %.2f ms\n", val, op_time);
		fprintf(stderr, "INFO: fill with val dram write %f performance: %.2f GB/s\n", val, 4 * n / (1e9 * op_time / 1e3));
#endif	

	}
	CUDA_ARRAY_SIZE_TYPE size() const {
		return n;
	}
	double download(T * h_d) {
		TimerGPU timer(0);
		safeCall(cudaMemcpy(data_ptr, h_d, sizeof(T) * (long long int)n, cudaMemcpyHostToDevice));
		double gpuTime = timer.read();
#ifdef VERBOSITY
		fprintf(stderr, "INFO: download time = %.2fms\n", gpuTime);
#endif
		return gpuTime;
	}	
	double readback(T * h_d) {
		TimerGPU timer(0);
		safeCall(cudaMemcpy(h_d, data_ptr, sizeof(T) * (long long int)n, cudaMemcpyDeviceToHost));
		double gpuTime = timer.read();
#ifdef VERBOSITY
		fprintf(stderr, "INFO: readback time = %.2fms\n", gpuTime);
#endif
		return gpuTime;
	}
	double download3D(T * h_d, const int nx, const int ny, const int nz, 
				   const int nxpad, const int nypad, const int nzpad);
	double readback3D(T * h_d, const int nx, const int ny, const int nz,
				   const int nxpad, const int nypad, const int nzpad);
public:
	CUDA_ARRAY_SIZE_TYPE n;
	T * data_ptr;
	int * ref_count;			
};

template<typename T>
double CudaArray<T>::download3D(T * h_d, const int nx, const int ny, const int nz,
					 const int nxpad, const int nypad, const int nzpad)
{
	TimerGPU timer(0);
	T * h_ptr = h_d;
	T * d_ptr = data_ptr;
	int h_stride_y = nx * nz;
	int d_stride_y = nxpad * nzpad;
	int stride_z = sizeof(T) * nzpad;
	for(int iy = 0; iy < ny; iy++) {
		safeCall(cudaMemcpy2D(d_ptr, stride_z, h_ptr, sizeof(T) * nz, sizeof(T) * nz, nx, cudaMemcpyHostToDevice));
		h_ptr += h_stride_y;
		d_ptr += d_stride_y;
	}
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: download time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	

}

template<typename T>
double CudaArray<T>::readback3D(T * h_d, const int nx, const int ny, const int nz,
					 const int nxpad, const int nypad, const int nzpad)
{
	TimerGPU timer(0);
	T * h_ptr = h_d;
	T * d_ptr = data_ptr;
	int h_stride_y = nx * nz;
	int d_stride_y = nxpad * nzpad;
	int stride_z = sizeof(T) * nzpad;
	for(int iy = 0; iy < ny; iy++) {
		safeCall(cudaMemcpy2D(h_ptr, sizeof(T) * nz, d_ptr, stride_z, sizeof(T) * nz, nx, cudaMemcpyDeviceToHost));
		h_ptr += h_stride_y;
		d_ptr += d_stride_y;
	}
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: readback time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	

}

template<size_t BX, size_t BY, size_t BZ, size_t TX, size_t TY, size_t TZ>
__global__ static void wf_update_gpu_kernel_template(float * __restrict__ wf_next, float * __restrict__ wf_cur, const int ny, const int nx, const int nz, float * __restrict__ vel, float * __restrict__ laplacian)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int tz = threadIdx.z;
	
	const int gx = blockIdx.x * BX;
	const int gy = blockIdx.y * BY;
	const int gz = blockIdx.z * BZ;
	
	const int stride_y = nx * nz;
	const int stride_y_k = nx * (nz / 2 + 1) * 2;
	const int stride_z_k = (nz / 2 + 1) * 2;

	const int inputIndex = gz * stride_y + gy * nz + gx;
	const int outputIndex = inputIndex;
	wf_cur += inputIndex;
	wf_next += outputIndex;
	vel += inputIndex;
	const int inputIndexLap = gz * stride_y_k + gy * stride_z_k + gx;
	laplacian += inputIndexLap;

	int ix, iy, iz;
	#pragma unroll
	for(iy = tz; iy < BZ; iy += TZ) {
		#pragma unroll
		for(ix = ty; ix < BY; ix += TY) {
			#pragma unroll
			for(iz = tx; iz < BX; iz += TX) {
				int index = iy * stride_y + ix * nz + iz;
				int indexK = iy * stride_y_k + ix * stride_z_k + iz;
				if(gz + iy < ny && gy + ix < nx && gx + iz < nz) wf_next[index] = wf_cur[index] + wf_cur[index] - wf_next[index] + vel[index] * laplacian[indexK];
//				if(gz + iy < ny && gy + ix < nx && gx + iz < nz) wf_next[index] = vel[index];
			}
		}
	}
}

__device__ float wsinc(float x)
{
	float sinc = 1.f;
	float wind = 0.f;
	float tmp = M_PI * x;
	if(x) sinc = __sinf(tmp) / tmp;
	if(x > -4.f && x < 4.f) wind = 0.5f * (1.f + __cosf(tmp / 4.f));
	return sinc * wind;
}

template<size_t BX, size_t BY, size_t BZ, size_t TX, size_t TY, size_t TZ>
__global__ static void inject_source_gpu_kernel_template(
	float * __restrict__ wf_cur, float wavelet, 
	float sx, float sy, float sz, 
	const int x1, const int x2, const int z1, const int z2, const int y1, const int y2, 
	const int ny, const int nx, const int nz)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int tz = threadIdx.z;
	
	const int gx = blockIdx.x * BX;
	const int gy = blockIdx.y * BY;
	const int gz = blockIdx.z * BZ;
	
	const int stride_y = nx * nz;

	const int inputIndex = gz * stride_y + gy * nz + gx;
	wf_cur += inputIndex;

	int ix, iy, iz;
	const float km = 0.68f;
	#pragma unroll
	for(iy = tz; iy < BZ; iy += TZ) {
		#pragma unroll
		for(ix = ty; ix < BY; ix += TY) {
			#pragma unroll
			for(iz = tx; iz < BX; iz += TX) {
				int z = gx + iz;
				int x = gy + ix;
				int y = gz + iy;
				if(x >= x1 && x < x2 && z >= z1 && z < z2 && y >= y1 && y < y2) {  
					float ys = wsinc(km * (sy - y));
					float xs = wsinc(km * (sx - x));
					float zs = wsinc(km * (sz - z));
					int index = iy * stride_y + ix * nz + iz;
					wf_cur[index] += wavelet * xs * ys * zs;
				}
			}
		}
	}
}

//TODO: fft_next_fast_size
inline int fft_next_fast_size(int n)
{
	while(1) {
		int m=n;
		while ( (m%2) == 0 ) m/=2;
		while ( (m%3) == 0 ) m/=3;
		while ( (m%5) == 0 ) m/=5;
		while ( (m%7) == 0 ) m/=7;
		if (m<=1)
		    break; /* n is completely factorable by twos, threes, fives and sevens */
		n++;
    	}
	return n;
}

float * get_lap3d_operator(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz)
{
	const int nkz = nz / 2 + 1;
	const float dkx = 2.f * M_PI / (nx * dx);
	const float dky = 2.f * M_PI / (ny * dy);
	const float dkz = 2.f * M_PI / (nz * dz);
	
	unsigned long long int msize = (unsigned long long)nx * ny * nkz;

	float * modu = (float*)malloc(sizeof(float) * msize);

	const int ThreadNum = omp_get_max_threads();
	const int nxkz = nx * nkz;
	const float scale = 1.f / (nx * ny * nz);

	#pragma omp parallel for num_threads(ThreadNum) schedule(dynamic)
	for(int iy = 0; iy < ny; iy++) {
		float * m_ptr = modu + iy * nxkz;
		float ky = iy < ny / 2 + 1 ? iy * dky : (iy - ny) * dky;
		float ky2 = ky * ky;
		for(int ix = 0; ix < nx; ix++) {
			float kx = ix < nx / 2 + 1 ? ix * dkx : (ix - nx) * dkx;
			float kxy2 = ky2 + kx * kx;
			for(int iz = 0; iz < nkz; iz++) {
				float kz = iz * dkz;
				float kmodu = kxy2 + kz * kz;
				*(m_ptr++) = - kmodu * scale; 
			}
		}
	}
	return modu;	
}

template<size_t BX, size_t BY, size_t BZ, size_t TX, size_t TY, size_t TZ>
double mod3d_psspec_gpu_wrapper(
	const int nx, const int ny, const int nz,
	const float dx, const float dy, const float dz,
	float * wf_cur, float * h_vel, 
	const int nt, const float dt, const float * wav, const int waveletLength)
{
	int nxNext = fft_next_fast_size(nx);
	int nyNext = fft_next_fast_size(ny);
	int nzNext = fft_next_fast_size(nz);
	int nkz = nzNext / 2 + 1;
	const CUDA_ARRAY_SIZE_TYPE nxykz = (CUDA_ARRAY_SIZE_TYPE)nxNext * nyNext * nkz;
	const CUDA_ARRAY_SIZE_TYPE nxyz = (CUDA_ARRAY_SIZE_TYPE)nxNext * nyNext * nzNext;

//	for(int iy = 0; iy < ny; iy++) {
//		for(int ix = 0; ix < nx; ix++) {
//			for(int iz = 0; iz < nz; iz++) {
//				wf_cur[(iy * nx + ix) * nz + iz] = sin(4.0 * M_PI * iz / nz);
//			}
//		}
//	}
		
	CudaArray<float> d_wf_cur(nxyz);
	d_wf_cur.fill(0.0);
	d_wf_cur.download3D(wf_cur, nx, ny, nz, nxNext, nyNext, nzNext);

	CudaArray<float> d_wf_next(nxyz);
	d_wf_next.fill(0.0);
	d_wf_next.download3D(wf_cur, nx, ny, nz, nxNext, nyNext, nzNext);

	CudaArray<Complex> d_laplacian(nxykz);
	
	CudaArray<float> d_modu(nxykz);
	float * h_modu = get_lap3d_operator(nxNext, nyNext, nzNext, dx, dy, dz);
	d_modu.download(h_modu); 
	free(h_modu); h_modu = nullptr;	

	CudaArray<float> d_vel(nxyz);
	d_vel.fill(h_vel[0]);
	d_vel.download3D(h_vel, nx, ny, nz, nxNext, nyNext, nzNext);

	/* create 3D FFT plans */
	cufftHandle r2c;
	cufftHandle c2r;
	int n[3] = {nyNext, nxNext, nzNext};
	int inembed[3] = {nyNext, nxNext, nzNext};
	int onembed[3] = {nyNext, nxNext, nkz};
	int ionembed[3] = {nyNext, nxNext, 2 * nkz};
	const int Nrank = 3;
	const int Batch = 1;
	if (cufftPlanMany(&r2c, Nrank, n, 
					  inembed, 1, nxyz, // *inembed, istride, idist
					  onembed, 1, nxykz, // *onembed, ostride, odist
					  CUFFT_R2C, Batch) != CUFFT_SUCCESS) {
		fprintf(stderr, "ERROR: Plan creation failed in file <%s>, line %i.\n", __FILE__, __LINE__);
	}

	if (cufftPlanMany(&c2r, Nrank, n, 
					  nullptr, 1, nxykz, // *inembed, istride, idist
					  nullptr, 1, 2 * nxykz, // *onembed, ostride, odist
					  CUFFT_C2R, Batch) != CUFFT_SUCCESS) {
		fprintf(stderr, "ERROR: Plan creation failed in file <%s>, line %i.\n", __FILE__, __LINE__);
	}

//	if (cufftExecR2C(r2c, (cufftReal*)d_wf_cur.data_ptr, (cufftComplex*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
//		fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
//	}
//
//	d_modu.fill(1.f / (nxyz));
//	d_laplacian = d_laplacian * d_modu;
//
//	if (cufftExecC2R(c2r, (cufftComplex*)d_laplacian.data_ptr, (cufftReal*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
//		fprintf(stderr, "ERROR: ExecC2R failed in file <%s>, line %i.\n", __FILE__, __LINE__);
//	}
//	
//	float * temp = (float*)malloc(sizeof(float) * 2 * nxykz);
//	d_laplacian.readback3D((cufftComplex*)temp, nx, ny, nz / 2 + 1, nxNext, nyNext, nkz);
//	
//	FILE * fp = fopen("/d0/data/zx/wf_gpu_spec.dat", "wb");
//	fwrite(temp, sizeof(float), 2 * nx * ny * (nz / 2 + 1), fp);
//	fflush(fp);
//	fclose(fp);
//
//	return 1.0;

	long long int mpoints = nxyz; 
	
	cout << 1.0 * mpoints * nt / 1e6 << "Mpoints\n";	
	
	float gflopsFFT = 2.f * nxyz * logf(nxyz) / 1e9;
	
	cout << 1.0 * gflopsFFT << "GFLOP\n";

	float gflopsWavNumOps = 2 * nxykz / 1e9; 
	
	cout << 1.0 * gflopsWavNumOps << "GFLOP\n";

	float gflopsUpdate = 4 * nxyz / 1e9; 
	
	cout << 1.0 * gflopsUpdate << "GFLOP\n";

	float gflops = gflopsFFT + gflopsWavNumOps + gflopsUpdate;
	
	cout << 1.0 * gflops << "GFLOP\n";

	const float sx = (nx - 1) * dx * 0.5f / dx;
	const float sz = (nz - 1) * dz * 0.5f / dz;
	const float sy = (ny - 1) * dy * 0.5f / dy;
	
	const float wind = 0.05f;
	const int windx = (int)(wind / dx + 0.5f);
	const int windy = (int)(wind / dy + 0.5f);
	const int windz = (int)(wind / dz + 0.5f);	

	const int x1 = sx - windx;
	const int x2 = sx + windx + 1;
	const int z1 = sz - windz;
	const int z2 = sz + windz + 1;
	const int y1 = sy - windy;
	const int y2 = sy + windy + 1;

	fprintf(stderr, "nxNext = %d, nyNext = %d, nzNext = %d\n", nxNext, nyNext, nzNext);

	int grid_x = iDivUp(nzNext, BX);
	int grid_y = iDivUp(nxNext, BY);
	int grid_z = iDivUp(nyNext, BZ);
	dim3 grids(grid_x, grid_y, grid_z);
	dim3 threads(TX, TY, TZ);

	TimerGPU timer(0);
	double timeFFT = 0.0;
	double timeWavNumOps = 0.0;
	double timeUpdate = 0.0;
	// kernel execution
	int it = 0;
	for(; it <= nt - 2; it += 2) {
		// from wf_cur -> wf_next
		if(it % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it, nt);
	
		TimerGPU t0(0);	
		if (cufftExecR2C(r2c, (cufftReal*)d_wf_cur.data_ptr, (cufftComplex*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
			fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
		}
		timeFFT += t0.read();

		TimerGPU t1(0);
		d_laplacian = d_laplacian * d_modu;
		timeWavNumOps += t1.read();

		TimerGPU t3(0);	
		if (cufftExecC2R(c2r, (cufftComplex*)d_laplacian.data_ptr, (cufftReal*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
			fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
		}
		timeFFT += t3.read();

		TimerGPU t4(0);
		wf_update_gpu_kernel_template<BX, BY, BZ, TX, TY, TZ><<<grids, threads>>>(
			d_wf_next.data_ptr, d_wf_cur.data_ptr, nyNext, nxNext, nzNext, 
			d_vel.data_ptr, (float*)d_laplacian.data_ptr);
		timeUpdate += t4.read();
		
		if(it < waveletLength) {
			inject_source_gpu_kernel_template<BX, BY, BZ, TX, TY, TZ><<<grids, threads>>>(
				d_wf_next.data_ptr, wav[it], 
				sx, sy, sz, 
				x1, x2, z1, z2, y1, y2, 
				nyNext, nxNext, nzNext);
		}
		
		// from wf_next -> wf_cur
		if((it + 1) % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it + 1, nt);
		
		TimerGPU t5(0);
		if (cufftExecR2C(r2c, (cufftReal*)d_wf_next.data_ptr, (cufftComplex*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
			fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
		}
		timeFFT += t5.read();

		TimerGPU t6(0);
		d_laplacian = d_laplacian * d_modu;
		timeWavNumOps += t6.read();
		
		TimerGPU t7(0);
		if (cufftExecC2R(c2r, (cufftComplex*)d_laplacian.data_ptr, (cufftReal*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
			fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
		}
		timeFFT += t7.read();

		TimerGPU t8(0);
		wf_update_gpu_kernel_template<BX, BY, BZ, TX, TY, TZ><<<grids, threads>>>(
			d_wf_cur.data_ptr, d_wf_next.data_ptr, nyNext, nxNext, nzNext, 
			d_vel.data_ptr, (float*)d_laplacian.data_ptr);
		timeUpdate += t8.read();
		
		if(it + 1 < waveletLength) {
			inject_source_gpu_kernel_template<BX, BY, BZ, TX, TY, TZ><<<grids, threads>>>(
				d_wf_cur.data_ptr, wav[it + 1], 
				sx, sy, sz, 
				x1, x2, z1, z2, y1, y2, 
				nyNext, nxNext, nzNext);
		}
	}
		
	for(; it < nt; it++) {
		if(it % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it, nt);
		
		TimerGPU t0(0);	
		if (cufftExecR2C(r2c, (cufftReal*)d_wf_cur.data_ptr, d_laplacian.data_ptr) != CUFFT_SUCCESS) {
			fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
		}
		timeFFT += t0.read();

		TimerGPU t1(0);
		d_laplacian = d_laplacian * d_modu;
		timeWavNumOps += t1.read();
		
		TimerGPU t3(0);
		if (cufftExecC2R(c2r, d_laplacian.data_ptr, (cufftReal*)d_laplacian.data_ptr) != CUFFT_SUCCESS) {
			fprintf(stderr, "ERROR: ExecR2C failed in file <%s>, line %i.\n", __FILE__, __LINE__);
		}
		timeFFT += t3.read();

		TimerGPU t4(0);
		wf_update_gpu_kernel_template<BX, BY, BZ, TX, TY, TZ><<<grids, threads>>>(
			d_wf_next.data_ptr, d_wf_cur.data_ptr, nyNext, nxNext, nzNext, 
			d_vel.data_ptr, (float*)d_laplacian.data_ptr);
		timeUpdate += t4.read();		

		if(it < waveletLength) {
			inject_source_gpu_kernel_template<BX, BY, BZ, TX, TY, TZ><<<grids, threads>>>(
				d_wf_next.data_ptr, wav[it], 
				sx, sy, sz, 
				x1, x2, z1, z2, y1, y2, 
				nyNext, nxNext, nzNext);
		}
	}		
	double gpuTime = timer.read();

	fprintf(stdout, "INFO: elapsed time = %.2f ms.\n", gpuTime);
	fprintf(stdout, "INFO: elapsed time of fft = %.2f ms.\n", timeFFT);
	fprintf(stdout, "INFO: elapsed time of wavenumber domain operator = %.2f ms.\n", timeWavNumOps);
	fprintf(stdout, "INFO: elapsed time of wavefield update = %.2f ms.\n", timeUpdate);
#ifdef VERBOSITY
	fprintf(stdout, "INFO: OVERALL performance = %.2f Mpoints/s.\n", mpoints * nt / (1e6 * gpuTime * 1e-3));
	fprintf(stdout, "INFO: OVERALL arithmetic operation throuput performance = %.2f GFLOPS.\n", gflops * nt / (gpuTime * 1e-3));
	fprintf(stdout, "INFO: performance of fft = %.2f GFLOPS.\n", gflopsFFT * nt / (timeFFT * 1e-3));
	fprintf(stdout, "INFO: arithmetic operations throuput of wavenumber domain operator = %.2f GFLOPS.\n", gflopsWavNumOps * nt / (timeWavNumOps * 1e-3));
	fprintf(stdout, "INFO: dram read performance of wavenumber domain operator = %.2f GB/s.\n", 2.f * 8 * nxykz * nt / (1e9 * timeWavNumOps * 1e-3));
	fprintf(stdout, "INFO: dram write performance of wavenumber domain operator = %.2f GB/s.\n", 8 * nxykz * nt / (1e9 * timeWavNumOps * 1e-3));
	fprintf(stdout, "INFO: arithmetic operations throuput of wavefield update = %.2f GFLOPS.\n", gflopsUpdate * nt / (timeUpdate * 1e-3));
	fprintf(stdout, "INFO: dram read performance of wavefield update = %.2f GB/s.\n", (2 * nxyz * 4 + nxyz * 4 + nxykz * 8) * nt / (1e9 * timeUpdate * 1e-3));
	fprintf(stdout, "INFO: dram write performance of wavfield update = %.2f GB/s.\n", 4 * nxyz * nt / (1e9 * timeUpdate * 1e-3));
#endif
	fflush(stdout);

	if((nt & 0x1) == 0) d_wf_cur.readback3D(wf_cur, nx, ny, nz, nxNext, nyNext, nzNext);
	else d_wf_next.readback3D(wf_cur, nx, ny, nz, nxNext, nyNext, nzNext);

#ifdef DEBUG
	FILE * fp = fopen("./wf_gpu_spec.dat", "wb");
	fwrite(wf_cur, sizeof(float), nx * ny * nz, fp);
	fflush(fp);
	fclose(fp);
#endif

	cufftDestroy(r2c);
	cufftDestroy(c2r);
	
	return gpuTime;
}

void set_source_wavelet(float * wav, const int nt, const float dt, const float f0)
{
	for(int it = 0; it < nt; it++) {
		float ttime = it * dt;
		float temp = M_PI * M_PI * f0 * f0 * (ttime - 1.0 / f0) * (ttime - 1.0 / f0);
		wav[it] = (1.0 - 2.0 * temp) * expf(- temp);
	}	
}

int main(int argc, char * argv[])
{
//	if(argc != 3) {
//		fprintf(stdout, "USAGE: valA valB\n");
//		return EXIT_USAGE;
//	}
//	const CUDA_ARRAY_SIZE_TYPE n = 1.66 * 1e9 / 4;
//	fprintf(stderr, "%lld\n", n);
//	fprintf(stderr, "%f GBytes\n", (long long int)n * 4 * 3 / 1e9);
//	float va = atof(argv[1]);
//	float vb = atof(argv[2]);
//	CudaArray<float> a(n);
//	CudaArray<float> b(n);
//	a.fill(0.0);
//	a.fill(va);
//	b.fill(vb);
//	CudaArray<float> c;
//	c = a / b;
//	c = a + b;
//	c = a - b;
//	c = a * b;
//#ifdef DEBUG	
//	float * h_a = (float*)malloc(sizeof(float) * n);
//	float * h_b = (float*)malloc(sizeof(float) * n);
//	float * h_c = (float*)malloc(sizeof(float) * n);
//	a.readback(h_a);
//	b.readback(h_b);
//	c.readback(h_c);
//
//#ifndef FLT_MIN
//#define FLT_MIN 1e-6
//#endif
//	FILE * fout = fopen("/d0/data/zx/debug.txt", "w");
//	for(CUDA_ARRAY_SIZE_TYPE i = 0; i < n; i++) {
//		if(fabsf(h_a[i] - va) > FLT_MIN || fabsf(h_b[i] - vb) > FLT_MIN || fabsf(h_c[i] - va * vb) > FLT_MIN) fprintf(fout, "ERROR: a[%d] = %f, b[%d] = %f, c[%d] = %f\n", i, h_a[i], i, h_b[i], i, h_c[i]);
//	}
//	fflush(fout);
//	fclose(fout);
//	free(h_a);
//	free(h_b);
//	free(h_c);
//#endif

	if(argc != 9) {
		fprintf(stderr, "USAGE: nx ny nz dx dy dz nt dt\n");
		return -1;
	}
	
	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int nz = atoi(argv[3]);
	float dx = atof(argv[4]);
	float dy = atof(argv[5]);
	float dz = atof(argv[6]);
	int nt = atoi(argv[7]);
	float dt = atof(argv[8]);

	const float f0 = 20.f;
	const int waveletLength = 2 * (int)(1.f / f0 / dt + 0.5f);
	float * wav = (float*)malloc(sizeof(float) * waveletLength);
	set_source_wavelet(wav, waveletLength, dt, f0);
	for(int it = 0; it < waveletLength; it++) {
		wav[it] *= dt * dt / (dx * dy * dz);
	}
	
#ifdef VERBOSITY	
	fprintf(stdout, "INFO: pseudo spectral forward modeling\n");
	fprintf(stdout, "INFO: nx = %d, ny = %d, nz = %d\n", nx, ny, nz);
	fflush(stdout);
#endif

	float * wf_cur = (float*)malloc(sizeof(float) * nx * ny * nz);
	float * wf_next = (float*)malloc(sizeof(float) * nx * ny * nz);
	float * h_vel = (float*)malloc(sizeof(float) * nx * ny * nz);
	float * h_wf_gpu = (float*)malloc(sizeof(float) * nx * ny * nz);
	
	memset(wf_cur, 0, sizeof(float) * nx * ny * nz);
	memset(wf_next, 0, sizeof(float) * nx * ny * nz);
	memset(h_wf_gpu, 0, sizeof(float) * nx * ny * nz);
	for(long long int i = 0; i < nx * ny * nz; i++) h_vel[i] = 3.f * 3.f * dt * dt;

	mod3d_psspec_gpu_wrapper<BLOCK_X, BLOCK_Y, BLOCK_Z, THREAD_X, THREAD_Y, THREAD_Z>(
		nx, ny, nz, 
		dx, dy, dz, 
		h_wf_gpu, h_vel, 
		nt, dt, wav, waveletLength); 	

	free(wf_cur); wf_cur = nullptr;
	free(wf_next); wf_next = nullptr;
	free(h_vel); h_vel = nullptr;
	free(h_wf_gpu); h_wf_gpu = nullptr;
	free(wav); wav = nullptr;

	return EXIT_SUCCESS;
}


