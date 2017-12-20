#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include "cuPrintf.cu"

#define INTEL_C_COMPILER
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
#define nullptr NULL

#define BSIZE 512
#define TSIZE 128
#define DEBUG

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

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

template<typename T>
class ExpressionUnary {
public:
	ExpressionUnary(CudaArray<T> __id) : id(__id) { };
	~ExpressionUnary() { };
	virtual void setOp() = 0;
public:
	CudaArray<T> id;
	T (*op)(T& a);
};

template<typename T>
class ExpressionBinary {
public:
	ExpressionBinary(CudaArray<T> __left, CudaArray<T> __right) : left(__left), right(__right) { };
	~ExpressionBinary() { };
	virtual void setOp() = 0;
public:
	CudaArray<T> left;
	CudaArray<T> right;
	T (*op)(T& a, T& b);
};

template<typename T>
class ExpressionAdd : public ExpressionBinary {
public:
	static T plus(T& a, T& b) {
		return a + b;
	}
	virtual void setOp() {
		op = &plus;
	}
}

template<typename T>
ExpressionAdd& operator+(CudaArray<T>& left, CudaArray<T>& right) {
	return ExpressionAdd(left, right);
}

template<typename T>
class CudaArray {
public:
	CudaArray() : n(0), data_ptr(nullptr), ref_count(nullptr) { 
#ifdef DEBUG
		fprintf(stderr, "INFO: default constructor of CudaArray<T> is called\n");
#endif
	}
	CudaArray(const int __n) : n(__n) {
#ifdef DEBUG
		fprintf(stderr, "INFO: constructor CudaArray(const int n) is called\n");
#endif
		if(n < 0) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from device memory, file: %s, line: %d\n", (long long int)n * sizeof(T), __FILE__, __LINE__);
			data_ptr = nullptr;
			ref_count = nullptr;
		} else {
			safeCall(cudaMalloc((void**)&data_ptr, n * sizeof(T)));
			if(data_ptr == nullptr) {
				fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", (long long int)n * sizeof(T), __FILE__, __LINE__);	
			}
			ref_count = new int(1);
		}
	}
	CudaArray(const CudaArray& arr) : n(arr.n) {
#ifdef DEBUG
		fprintf(stderr, "INFO: copy constructor of CudaArray<T> is called\n");
#endif
		data_ptr = arr.data_ptr;
		ref_count = arr.ref_count;
		if(ref_count != nullptr) *ref_count += 1;
	}
	CudaArray& operator=(const CudaArray& arr) {
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
					ref_count == nullptr;
				}
			}
			data_ptr = arr.data_ptr;
			ref_count = arr.ref_count;
			if(ref_count != nullptr) *ref_count += 1;
		}
		return *this;
	}
	CudaArray(CudaArray&& temp) : n(temp.n) {
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
	CudaArray& operator=(CudaArray&& temp) {
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
	~CudaArray() {
#ifdef DEBUG
		fprintf(stderr, "INFO: deconstructor of CudaArray<T> is called\n");
#endif
		if(ref_count != nullptr) {
			if(*ref_count > 1) *ref_count -= 1;
			else if(*ref_count == 1) {
				delete ref_count;
				ref_count = nullptr;
				safeCall(cudaFree(data_ptr));
				data_ptr = nullptr;
			}
		}
	}
	CudaArray& operator=(const ExpressionBinary& bin_op) {
#ifdef DEBUG
		fprintf(stderr, "INFO: assignment operator =(ExpressionBinary&) of CudaArray<T> is called\n");
#endif
		assert(bin_op.left.size() == bin_op.right.size());
		const int osize = bin_op.left.size();
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
			safeCall(cudaMalloc((void**)&data_ptr, n * sizeof(T)));
			ref_count = new int(1);
			if(data_ptr == nullptr) {
				fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", (long long int)n * sizeof(T), __FILE__, __LINE__);
			}
		}
		int threads = TSIZE;
		int blocks = iDivUp(n, BSIZE);
		TimerGPU timer(0.0);
		cuda_array_binary_op<BSIZE, TSIZE><<<threads, blocks>>>(data_ptr, bin_op.left.data_ptr, bin_op.right.data_ptr, n, bin_op.op);
		double op_time = timer.read();
#ifdef DEBUG
		fprintf(stderr, "INFO: binary operator gpu time %.2f\n", op_time);
#endif	
		return *this;	
	}
	void fill(T& val) {
		int threads = TSIZE;
		int blocks = iDivUp(n, BSIZE);
		TimerGPU timer(0.0);
		cuda_array_fill<BSIZE, TSIZE><<<threads, blocks>>>(data_ptr, val);
		double op_time = timer.read();
#ifdef DEBUG
		fprintf(stderr, "INFO: fill with val %f gpu time %.2f\n", val, op_time);
#endif	

	}
	template<size_t BSIZE, size_t TSIZE, typename T> __global__ static void cuda_array_binary_op(T * od, const T * __left, const T * __right, const int n, T (*op)(T& a, T& b)) {
		const int tx = threadIdx.x;
		const int gx = blockIdx.x * BSIZE;
		
		int i;
		#pragma unroll
		for(i = tx; i < BSIZE; i += TSIZE) {
			if(i < n) od[i] = bin_op(__left[i], __right[i]);
		}
	}
	template<size_t BSIZE, size_t TSIZE, typename T> __global__ static void cuda_array_fill(T * od, T& val) {
		const int tx = threadIdx.x;
		const int gx = blockIdx.x * BSIZE;
		
		int i;
		#pragma unroll
		for(i = tx; i < BSIZE; i += TSIZE) {
			if(i < n) od[i] = val;
		}
	}
public:
	int n;
	T * data_ptr;
	int * ref_count;			
};

int main(int argc, char * argv[])
{
	const int n = 1e8;
	CudaArray<float> a(n);
	CudaArray<float> b(n);
	a.fill(1.0);
	b.fill(2.0);
	CudaArray<float> c = a + b;

	return EXIT_SUCCESS;	
}


