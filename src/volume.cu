#include "util.h"
#include "volume.h"

__global__ void smoothf(float *v, float *to, int std, int c, int X, int Y, int Z, int C) {
  int x(threadIdx.x + blockDim.x * blockIdx.x);
  int y(threadIdx.y + blockDim.y * blockIdx.y);
  int z(threadIdx.z + blockDim.z * blockIdx.z);
  if (x >= X || y >= Y || z >= Z) return;
  
  int index = z * C * X * Y + c * X * Y + y * X + x;
  int rad = std * 2;
  float sum(0), weight(0.000001);
  
  for (int zz(max(0, z - rad)); zz < min(zz + rad, Z); ++zz)
    //for (int cc(max(0, x - rad)); cc < min(cc + rad, C); ++c)
    for (int yy(max(0, y - rad)); yy < min(y + rad, Y); ++yy)
      for (int xx(max(0, x - rad)); xx < min(x + rad, X); ++xx) {
	int index = zz * C * X * Y + c * X * Y + yy * X + xx;
	float diffx = (static_cast<float>(x - xx) / std);
	float diffy = (static_cast<float>(y - yy) / std);
	float diffz = (static_cast<float>(z - zz) / std);
	float w = exp(-(diffx * diffx + diffy * diffy + diffz * diffz));
	sum += v[index] * w;
	weight += w;
      }
  
  to[index] = sum / weight;
}

void smooth(Volume &in, Volume &out, int std, int c) {
  int const BLOCKSIZE(32);
  VolumeShape s(in.shape);
  
  dim3 dimBlock( BLOCKSIZE, BLOCKSIZE, 1 );
  dim3 dimGrid( (s.w + BLOCKSIZE - 1) / BLOCKSIZE, (s.h + BLOCKSIZE - 1) / BLOCKSIZE, s.z);
  
  smoothf<<<dimGrid, dimBlock>>>(in.data(), out.data(), std, c, s.w, s.h, s.z, s.c);
  
}
