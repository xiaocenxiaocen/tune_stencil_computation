#include <mkl.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <time.h>
#include <iostream>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX

#include <cuda_runtime.h>

#include "gpu_mat_mul.h"

using namespace std;

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) > 0 ? (a) : (-(a)))

#define NUM_THREADS 12

template<typename T> T * ArrayAlloc(const size_t n0)
{
	T * ptr __attribute__((aligned(16))) = (T*)malloc(sizeof(T) * n0);
	memset(ptr, 0, sizeof(T) * n0);
	return ptr;
}

/* @brief: template function for 2d array allocation
 * @param: n0, n1, size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc2D(T ***buffer, const int n0, const int n1)
{
	*buffer = (T**)malloc(sizeof(T*) * n0);
	(*buffer)[0] = (T*)malloc(sizeof(T) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	return;
}

/* @brief: template function for 3d array allocation
 * @param: n0, n1, n2 size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc3D(T ****buffer, const int n0, const int n1, const int n2)
{
	*buffer = (T***)malloc(sizeof(T**) * n0);
	(*buffer)[0] = (T**)malloc(sizeof(T*) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	(*buffer)[0][0] = (T*)malloc(sizeof(T) * n0 * n1 * n2);
	for(int i1 = 1; i1 < n1; i1++) (*buffer)[0][i1] = (*buffer)[0][0] + i1 * n2;
	for(int i0 = 1; i0 < n0; i0++) {
		for(int i1 = 0; i1 < n1; i1++) {
			(*buffer)[i0][i1] = (*buffer)[0][0] + i0 * n1 * n2 + i1 * n2;
		}
	}
	return;
}

#define ArrayFree3D(ptr) \
{ \
	free(**(ptr)); \
	free(*(ptr)); \
	free(ptr); \
}

#define ArrayFree2D(ptr) \
{ \
	free(*(ptr)); \
	free(ptr); \
}

#define SCHEME_ORDER 8

#if SCHEME_ORDER == 8
static const float a[] __attribute__((aligned(16))) = { 0.8, -0.2, 0.038095238, -0.0035714286 };
static const float b[] __attribute__((aligned(16))) = { 1.6, -0.2, 0.02539682540, -0.001785714286 };
static const float b0 = -2.847222222;
#endif

inline float fd8_dx(const float *  wfl, const float dx, const int stride)
{
	return (a[0] * (*(wfl + 1 * stride) - *(wfl - 1 * stride)) + a[1] * (*(wfl + 2 * stride) - *(wfl - 2 * stride)) + a[2] * (*(wfl + 3 * stride) - *(wfl - 3 * stride)) + a[3] * (*(wfl + 4 * stride) - *(wfl - 4 * stride))) / dx;
}

inline float fd8_dxx(const float *  wfl, const float dx, const int stride)
{
	return (b0 * (*wfl) + b[0] * (*(wfl + 1 * stride) + *(wfl - 1 * stride)) + b[1] * (*(wfl + 2 * stride) + *(wfl - 2 * stride)) + b[2] * (*(wfl + 3 * stride) + *(wfl - 3 * stride)) + b[3] * (*(wfl + 4 * stride) + *(wfl - 4 * stride))) / (dx * dx);
}

void fdtd_naive(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float *** po, float *** pm, const int nt, const float dt)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;

	for(int it = 0; it < nt; it++) {
		po[sy * nx * nz + sx * nz + sz] += wav[it];		

		for(int iy = 0; iy < ny; iy++) {
			for(int ix = 0; ix < nx; ix++) {
				for(int iz = 0; iz < nz; iz++) {
					int idx = iy * nx * nz + ix * nz + iz;
					
					float pxx = fd8_dxx(po + idx, dx, nz);
					float pyy = fd8_dxx(po + idx, dy, nx * nz);
					float pzz = fd8_dxx(po + idx, dz, 1);					
					pm[idx] = 2.0f * po[idx] - pm[idx] + v * (pxx + pyy + pzz);
				}
			}
		}
		
		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}
}

void fdtd_openmp_naive(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float *** po, float *** pm, const int nt, const float dt)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;

	for(int it = 0; it < nt; it++) {
		po[sy * nx * nz + sx * nz + sz] += wav[it];		

		#ifdef _OPENMP
		#pragma omp parallel for num_threads(NUM_THREADS) shared(po, pm, nx, ny, nz, dx, dy, dz) schedule(dynamic)
		#endif
		for(int iy = 0; iy < ny; iy++) {
			for(int ix = 0; ix < nx; ix++) {
				for(int iz = 0; iz < nz; iz++) {
					int idx = iy * nx * nz + ix * nz + iz;
					
					float pxx = fd8_dxx(po + idx, dx, nz);
					float pyy = fd8_dxx(po + idx, dy, nx * nz);
					float pzz = fd8_dxx(po + idx, dz, 1);					
					pm[idx] = 2.0f * po[idx] - pm[idx] + v * (pxx + pyy + pzz);
				}
			}
		}
		
		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}
}

void fdtd_openmp_naive(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float *** po, float *** pm, const int nt, const float dt)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;
	const float va0 = v * a0 / (dx * dx) + v * a0 / (dy * dy) + v * a0 / (dz * dz);

#define CACHE_LINE_SIZE 64
	const int thread_block_size = CACHE_LINE_SIZE / sizeof(float) - SCHEME_ORDER;
	for(int it = 0; it < nt; it++) {
		po[sy][sx][sz] += wav[it];		

		for(int iy = SCHEME_ORDER / 2; iy < ny - SCHEME_ORDER / 2; iy++) {
			#ifdef _OPENMP
			#pragma omp parallel for num_threads(NUM_THREADS) shared(po, pm, nx, ny, nz, dx, dy, dz) schedule(dynamic)
			#endif
			for(int ix = SCHEME_ORDER/2; ix < nx - SCHEME_ORDER/2; ix += thread_block_size) {
				for(int iz = SCHEME_ORDER/2; iz < nz - SCHEME_ORDER/2; iz += thread_block_size) {
					float * po_ptr = po[iy][ix];
					float * pm_ptr = pm[iy][ix];
					for(int iix = 0; iix < thread_block_size && ix + iix < nx - SCHEME_ORDER/2; iix++, po_ptr += nz, pm_ptr += nz) {
						for(int iiz = 0; iiz < thread_block_size && iz + iiz < nz - SCHEME_ORDER/2; iiz++) {
							pm_ptr[iiz] = ( 2.0f + va0 ) * po_ptr[iiz] - pm_ptr[iiz];
						}
					}
					float * pxx_ptr = po[iy][ix - SCHEME_ORDER/2];
					float * pyy_ptr = po[iy - SCHEME_ORDER/2][ix];
					for(int ii = -SCHEME_ORDER; ii <= SCHEME_ORDER; ii++, pyy_ptr += nx * nz, pxx_ptr += nz) {
						for(int iix = 0; iix < thread_block_size && ix + iix < nx - SCHEME_ORDER/2; iix++, po_ptr += nz, pm_ptr += nz) {
							for(int iiz = 0; iiz < thread_block_size && iz + iiz < nz - SCHEME_ORDER/2; iiz++) {
								pm_ptr[iiz] = 2.0f * po_ptr[iiz] - pm_ptr[iiz];
							}
						}
					}
				}
			}
		}
		
		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}
}
