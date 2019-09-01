#ifndef __VOLUME_H__
#define __VOLUME_H__

#include <vector>
#include <map>
#include <string>
#include <cassert>
#include "tensor.h"
#include "cudavec.h"

//initially just float
typedef float F;

struct VolumeShape {
	int z, c, w, h;

	int size() const;
	int offset(int z, int c, int x, int y) const;
	int offsetrm(int z, int c, int x, int y) const;
  int slice_size() const;
};

std::ostream &operator<<(std::ostream &out, VolumeShape shape);

struct Volume {
	Volume(VolumeShape shape = VolumeShape{0, 0, 0, 0});
	Volume(VolumeShape shape, Volume &reuse_buffer);

  //Volume(Volume&& o) noexcept : shape(o.shape),  buf(o.buf), slice_size(o.slice_size), reused(o.reused) { }
  //  Volume(Volume& o) noexcept : shape(o.shape),  buf(o.buf), slice_size(o.slice_size), reused(o.reused) { }
  //Volume(Volume const &o);
  Volume (Volume && o) : shape(o.shape),  buf(o.buf), slice_size(o.slice_size), reused(o.reused) { }
  Volume &operator=(Volume && o) {
    shape = o.shape;
    buf = o.buf;
    slice_size = o.slice_size;
    reused = o.reused;
  }

    Volume (Volume & o) : shape(o.shape), slice_size(o.slice_size), reused(o.reused), buf(new CudaVec(0)) {
      from_volume(o);
    }
  
  Volume &operator=(Volume & o) {
    shape = o.shape;
    buf = new CudaVec(0);
    slice_size = o.slice_size;
    reused = o.reused;
    from_volume(o);
  }

  Volume(std::string filename);
  //Volume(Volume &&o);
  //	Volume &operator=(const Volume&);
	~Volume();

	float *slice(int z);
	TensorShape slice_shape();
	void zero(int offset = 0);
        void rand_zero(float p);
	void init_normal(F mean, F std);
        void add_normal(F mean, F std);
	void fill(F val);
  	int size();
  	float norm();
  	float norm2();

  	void from_volume(Volume &other);
  	std::vector<F> to_vector();
  	void from_vector(std::vector<F> &vec);
  void reshape(VolumeShape s);
  
  void thresholding(std::vector<F> &data, float threshold);
	void draw_slice(std::string filename, int slice, int channel = 0);
  void draw_volume(std::string filename, int channel = 0);
	void draw_slice_rgb(std::string filename, int slice);
	void dropout(float p);
	float *data();

  void save_file(std::string filename);
  void load_file(std::string filename);

  void insert_volume_at_c(Volume &other, int c);
  
	VolumeShape shape;
	CudaVec *buf;
	int slice_size;
	bool reused;
};

Volume &operator-=(Volume &in, Volume &other);
__global__ void smoothf(float *v, float *to, int std, int c, int X, int Y, int Z, int C);
void smooth(Volume &in, Volume &out, int std, int c = 0);

Volume join_volumes(std::vector<Volume*> volumes);

struct VolumeSet {
	VolumeSet(VolumeShape shape);
	VolumeSet(VolumeShape shape, VolumeSet &reuse_buffer);
	void zero();

	Volume x, diff;	// x: activation
	VolumeShape shape;
};

typedef std::map<std::string, VolumeSet*> VolumeSetMap;

// struct Volume6DSet {
// 	// order: [x,y,z], [y,x,z], [x, z, y]
// 	//

// 	Volume6DSet(VolumeShape shape);
// 	void zero();

// 	std::vector<VolumeSet*> volumes;
// 	std::vector<Volume*> x, diff;
// 	VolumeShape shape;
// };


#endif
