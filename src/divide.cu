#include "divide.h"
#include "util.h"
#include "rand.h"
//#include "util/cuPrintf.cuh"
// #include "util/cuPrintf.cu"

using namespace std;

__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, float *out, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;

	int in_index = get_index(X, Y, Z, C, x, y, z);

	switch(n){
		case ZF:
			copy_c(in + in_index, out + get_index(X, Y, Z, C, x, y,         z), X * Y, X * Y, C);
			break;
		case ZB:
			copy_c(in + in_index, out + get_index(X, Y, Z, C, x, y, Z - 1 - z), X * Y, X * Y, C);
			break;
		case XF:
			copy_c(in + in_index, out + get_index(Z, Y, X, C, z, y,         x), X * Y, Z * Y, C);
			break;
		case XB:
			copy_c(in + in_index, out + get_index(Z, Y, X, C, z, y, X - 1 - x), X * Y, Z * Y, C);
			break;
		case YF:
			copy_c(in + in_index, out + get_index(X, Z, Y, C, x, z,         y), X * Y, X * Z, C);
			break;
		case YB:
			copy_c(in + in_index, out + get_index(X, Z, Y, C, x, z, Y - 1 - y), X * Y, X * Z, C);
			break;
		// default:
		// 	throw "";
	}
}

__global__ void combine_kernel(int X, int Y, int Z, int C, float *in, float const *out, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;

	int in_index = get_index(X, Y, Z, C, x, y, z);

	switch(n){
		case ZF:
			add_c(out + get_index(X, Y, Z, C, x, y,         z), in + in_index, X * Y, X * Y, C);
			break;
		case ZB:
			add_c(out + get_index(X, Y, Z, C, x, y, Z - 1 - z), in + in_index, X * Y, X * Y, C);
			break;
		case XF:
			add_c(out + get_index(Z, Y, X, C, z, y,         x), in + in_index, Z * Y, X * Y, C);
			break;
		case XB:
			add_c(out + get_index(Z, Y, X, C, z, y, X - 1 - x), in + in_index, Z * Y, X * Y, C);
			break;
		case YF:
			add_c(out + get_index(X, Z, Y, C, x, z,         y), in + in_index, X * Z, X * Y, C);
			break;
		case YB:
			add_c(out + get_index(X, Z, Y, C, x, z, Y - 1 - y), in + in_index, X * Z, X * Y, C);
			break;
	}

}

__global__ void copy_subvolume_kernel(VolumeShape inshape, VolumeShape outshape, float *in, float *out,
	VolumeShape in2shape, VolumeShape out2shape,
	float *in2, float *out2, int xs, int ys, int zs, bool xflip, bool yflip, bool zflip, float deg, bool *succ){

	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= outshape.w || y >= outshape.h || z >= outshape.z)
		return;

	/// rotation
	int xnew(0), ynew(0);
	if (deg != 0.0){
		int xc = inshape.w - inshape.w/2;
	    int yc = inshape.h - inshape.h/2;

	    xnew = ((float)(xs+x)-xc)*cos(deg) - ((float)(ys+y)-yc)*sin(deg) + xc;
	    ynew = ((float)(xs+x)-xc)*sin(deg) + ((float)(ys+y)-yc)*cos(deg) + yc;

	    if (!*succ || !(xnew >= 0 && xnew < inshape.w && ynew >= 0 && ynew < inshape.h) ) {
	    	*succ = false;
	    	return;
	    }
	}
	else{
		xnew = xs+x;
		ynew = ys+y;
	}

	int outx = xflip ? (outshape.w - 1 - x) : x;
	int outy = yflip ? (outshape.h - 1 - y) : y;
	int outz = zflip ? (outshape.z - 1 - z) : z;

	int in_index = get_index(inshape.w, inshape.h, inshape.z, inshape.c, xnew, ynew, z+zs);
	int in2_index = get_index(in2shape.w, in2shape.h, in2shape.z, in2shape.c, xnew, ynew, z+zs);
	int out_index = get_index(outshape.w, outshape.h, outshape.z, outshape.c, outx, outy, outz);
	int out2_index = get_index(out2shape.w, out2shape.h, out2shape.z, out2shape.c, outx, outy, outz);

	copy_c(in + in_index, out + out_index, inshape.w * inshape.h, outshape.w * outshape.h, outshape.c);
	copy_c(in2 + in2_index, out2 + out2_index, in2shape.w * in2shape.h, out2shape.w * out2shape.h, out2shape.c);
}

__global__ void copy_subvolume_test_kernel(VolumeShape inshape, VolumeShape outshape, float *in, float *out, int xs, int ys, int zs) {

	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= outshape.w || y >= outshape.h || z >= outshape.z)
		return;

	int outx = x;
	int outy = y;
	int outz = z;
	int in_index = get_index(inshape.w, inshape.h, inshape.z, inshape.c, x+xs, y+ys, z+zs);
	int out_index = get_index(outshape.w, outshape.h, outshape.z, outshape.c, outx, outy, outz);

	copy_c(in + in_index, out + out_index, inshape.w * inshape.h, outshape.w * outshape.h, outshape.c);
}


void divide(Volume &from, Volume &to, int n) {
	VolumeShape shape = from.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w + BW - 1) / BW, (shape.h + BH - 1) / BH, shape.z );
	// cout << "Divide shape: " << shape << " " << to.shape << endl;
	// cout << ((shape.w + BW - 1) / BW) << " " <<  ((shape.h + BH - 1) / BH) << " " <<  shape.z << endl;
	divide_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, from.data(), to.data(), n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	// cout << "done" << endl;
}

void combine(Volume &from, Volume &to, int n) {
	VolumeShape shape = to.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w + BW - 1) / BW, (shape.h + BH - 1) / BH, shape.z );

	combine_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, to.data(), from.data(), n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}

void copy_subvolume(Volume &in, Volume &out, Volume &in2, Volume &out2, bool rotate, bool xflip, bool yflip, bool zflip) {
	VolumeShape inshape = in.shape;
	VolumeShape outshape = out.shape;
	VolumeShape in2shape = in2.shape;
	VolumeShape out2shape = out2.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (outshape.w + BW - 1) / BW, (outshape.h + BH - 1) / BH, outshape.z );

	bool succeed = false;
	while (!succeed){
	        int x = Rand::randn(in.shape.w - out.shape.w + 1);
		int y = Rand::randn(in.shape.h - out.shape.h + 1);
		int z = Rand::randn(in.shape.z - out.shape.z + 1);
		float deg = rotate ? rand_float() * 3.14 * .5 : 0;

		cout <<"copy_subvolume-idx: " << x << " " << y << " " << z <<  " / deg: " << deg << endl;
		bool *succ;
		cudaMalloc( (void**)&succ, sizeof(int) );
		cudaMemset( (void*)succ, 1, sizeof(int) );
		copy_subvolume_kernel<<<dimGrid, dimBlock>>>(inshape, outshape, in.data(), out.data(),
			in2shape, out2shape, in2.data(), out2.data(), x, y, z, xflip, yflip, zflip, deg, succ);
		cudaMemcpy( &succeed, succ, sizeof(bool), cudaMemcpyDeviceToHost );
		cudaFree( succ );
		cout << "	rotation succeed: " << succeed << "\n";
		if (!rotate)
		  break;
		if (!succeed)
			rotate = false;


	}

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());

}

void copy_subvolume_test(Volume &in, Volume &out, int stx, int sty, int stz) {
	VolumeShape inshape = in.shape;
	VolumeShape outshape = out.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (outshape.w + BW - 1) / BW, (outshape.h + BH - 1) / BH, outshape.z );

	int x = stx;
	int y = sty;
	int z = stz;
	cout <<"copy_subvolume_test-idx " << x << " " << y << " " << z << endl;

	copy_subvolume_test_kernel<<<dimGrid, dimBlock>>>(inshape, outshape, in.data(), out.data(), x, y, z);

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}
