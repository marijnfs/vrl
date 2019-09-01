#ifndef __DIVIDE_H__
#define __DIVIDE_H__

#include <cuda.h>
#include "volume.h"

enum Direction {
	ZF, // X/Y/B - FORWARD/BACKWARD
	ZB,
	XF,
	XB,
	YF,
	YB
};

__device__ __forceinline__ int get_index(int X, int Y, int Z, int C, int x, int y, int z) {
	// return z * (C * X * Y) + x * Y + y; //CWH, as cudnn
	return z * (C * X * Y) + y * X + x;//CHW, what what i thinking
}

__device__ __forceinline__ void copy_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
	// C = 1;
	for (size_t c(0); c < C; ++c)
		out[c * slicesizeout] = in[c * slicesizein];
}

__device__ __forceinline__ void add_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
	for (size_t c(0); c < C; ++c)
		out[c * slicesizeout] += in[c * slicesizein];
}

__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, float *out, int n);

__global__ void combine_kernel(int X, int Y, int Z, int C, float *in, float const *out, int n);

__global__ void copy_subvolume_kernel(VolumeShape inshape, VolumeShape outshape, float *in, float *out, VolumeShape in2shape, VolumeShape out2shape,
	float *in2, float *out2, int xs, int ys, int zs, bool xflip, bool yflip, bool zflip, float deg, bool *succ);

__global__ void copy_subvolume_test_kernel(VolumeShape inshape, VolumeShape outshape, float *in, float *out, int xs, int ys, int zs);



void divide(Volume &v, Volume &to, int n);
void combine(Volume &v, Volume &to, int n);
void copy_subvolume(Volume &in, Volume &out, Volume &in2, Volume &out2, bool rotate = false, bool xflip=false, bool yflip=false, bool zflip=false);
void copy_subvolume_test(Volume &in, Volume &out, int stx=0, int sty=0, int stz=0);

#endif
