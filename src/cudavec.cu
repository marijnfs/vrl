#include "util.h"
#include "cudavec.h"

__global__ void clip_kernel(float *v, int n, float limit) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = (v[x] > limit) ? limit : ((v[x] < -limit) ? -limit : v[x]);
}

__global__ void sqrt_kernel(float *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = sqrt(v[x]);
}

__global__ void abs_kernel(float *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = abs(v[x]);
}

__global__ void pow_kernel(float *v, int n, float e) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = pow(v[x], e);
}

__global__ void exp_kernel(float *v, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] = exp(v[x]);
}

__global__ void times_kernel(float *v, float *other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] *= other[x];
}

__global__ void divide_kernel(float *v, float *other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] /= other[x];
}

__global__ void times_scalarf(float *v, float other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] *= other;
}

__global__ void add_scalarf(float *v, float other, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	if (x >= n) return;

	v[x] += other;
}

void CudaVec::rand_zero(float p) {
	::rand_zero(data, n, p);
}


CudaVec &CudaVec::sqrt() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	sqrt_kernel<<<dimGrid, dimBlock>>>(data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::clip(float limit) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	clip_kernel<<<dimGrid, dimBlock>>>(data, n, limit);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::abs() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	abs_kernel<<<dimGrid, dimBlock>>>(data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::pow(float e) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	pow_kernel<<<dimGrid, dimBlock>>>(data, n, e);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::exp() {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	exp_kernel<<<dimGrid, dimBlock>>>(data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::add(int idx, float val) {

	add_scalarf<<<1, 1>>>(data+idx, val, 1);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}


CudaVec &CudaVec::operator*=(CudaVec &other) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	times_kernel<<<dimGrid, dimBlock>>>(data, other.data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::operator/=(CudaVec &other) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	divide_kernel<<<dimGrid, dimBlock>>>(data, other.data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::operator-=(CudaVec &other) {
	add_cuda<float>(other.data, data, n, -1);
	return *this;
}

CudaVec &CudaVec::operator+=(CudaVec &other) {
	add_cuda<float>(other.data, data, n, 1);
	return *this;
}


CudaVec &CudaVec::operator*=(float v) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	times_scalarf<<<dimGrid, dimBlock>>>(data, v, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}

CudaVec &CudaVec::operator+=(float v) {
	//primitive blocksize determination
	int const BLOCKSIZE(1024);

	dim3 dimBlock( BLOCKSIZE );
	dim3 dimGrid( (n + BLOCKSIZE - 1) / BLOCKSIZE );

	add_scalarf<<<dimGrid, dimBlock>>>(data, v, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
	return *this;
}
