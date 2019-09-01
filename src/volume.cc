#include "volume.h"
#include "util.h"
#include "img.h"
#include "gzstream.h"

#include <cmath>

using namespace std;

Volume::Volume(VolumeShape shape_) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h), reused(false) {
	// size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	cout << "allocating volume: " << shape << " nfloats: " << size() << endl;
	//handle_error( cudaMalloc((void**)&data, sizeof(float) * size()) 	);
	buf = new CudaVec(size());
	zero();
}

Volume::Volume(VolumeShape shape_, Volume &reuse_buffer) : shape(shape_), buf(reuse_buffer.buf),
							   slice_size(shape_.c * shape_.w * shape_.h), reused(true)							   
{
	if (size() > reuse_buffer.buf->n) {
		cout << "resizing " << size() << " / "  << reuse_buffer.buf->n << endl;
		reuse_buffer.buf->resize(size());
	}
	//data = reuse_buffer.data;
	zero();
}

Volume::Volume(string filename) : buf(new CudaVec()), reused(false) {
  load_file(filename);
}

/*
Volume::Volume(Volume &&o) : shape(o.shape), buf(o.buf), slice_size(o.slice_size), reused(o.reused) {
}

Volume::Volume(Volume const &o) : shape(o.shape), buf(o.buf), slice_size(o.slice_size), reused(o.reused) {
}

Volume& Volume::operator=(Volume const &o) {
	shape = o.shape;
	buf = o.buf;
	slice_size = o.slice_size;
	reused  = o.reused;
	return *this;
}
*/

Volume::~Volume(){
  cout << "destruct volume" << endl;
  if (!reused) {
    cout << "deleting buf" << endl;
    delete buf;
  }
}

void Volume::reshape(VolumeShape s) {
  cout << "reshaping: " << s << endl;
  buf->resize(s.size());
  shape = s;
  slice_size = shape.slice_size();
}

float *Volume::slice(int z) {
	return buf->data + z * slice_size;
}

TensorShape Volume::slice_shape() {
	return TensorShape{1, shape.c, shape.w, shape.h};
}

void Volume::zero(int offset) {
	buf->zero(offset * slice_size);
}

void Volume::rand_zero(float p) {
	buf->rand_zero(p);
}

void Volume::save_file(string filename) {
  ogzstream out_file(filename.c_str());
  byte_write<int32_t>(out_file, shape.z);
  byte_write<int32_t>(out_file, shape.c);
  byte_write<int32_t>(out_file, shape.w);
  byte_write<int32_t>(out_file, shape.h);

  vector<float> data = to_vector();
  for (auto v : data)
    byte_write<float>(out_file, v);

}

void Volume::load_file(string filename) {
  //Read Header
  igzstream in_file(filename.c_str());
  shape.z = byte_read<int32_t>(in_file);
  shape.c = byte_read<int32_t>(in_file);
  shape.w = byte_read<int32_t>(in_file);
  shape.h = byte_read<int32_t>(in_file);

  //Setup Internal Parameters
  buf->resize(shape.size());
  slice_size = shape.slice_size();

  //Read Data
  vector<float> data(shape.size());
  for (auto &v : data)
    v = byte_read<float>(in_file);

  //To GPU
  from_vector(data);
}

void Volume::init_normal(F mean, F std) {
	// size_t even_size(((size() + 1) / 2) * 2);
	// zero();
	buf->add_normal(mean, std);
	// ::init_normal(data, size(), mean, std);
	// handle_error( curandGenerateNormal(Handler::curand(), data, even_size, mean, std) );
}

void Volume::add_normal(F mean, F std) {
	// size_t even_size(((size() + 1) / 2) * 2);
	// zero();
	buf->add_normal(mean, std);
	// ::init_normal(data, size(), mean, std);
	// handle_error( curandGenerateNormal(Handler::curand(), data, even_size, mean, std) );
}

void Volume::fill(F val) {
	throw StringException("not implemented");
}

void Volume::from_volume(Volume &other) {
	if (size() != other.size()) {
	  cout << shape << " " << other.shape << endl;
	  cout << shape.size() << " " << other.shape.size() << endl;
	  throw StringException("sizes don't match");
	}
	copy_gpu_to_gpu(other.buf->data, buf->data, size());
}

void Volume::insert_volume_at_c(Volume &other, int c) {
  assert(shape.w == other.shape.w);
  assert(shape.h == other.shape.h);
  assert(shape.z == other.shape.z);
  
  
  for (int slice(0); slice < shape.z; ++slice) {
    float *to = data() + shape.offset(slice, c, 0, 0);
    float *from = other.data() + other.shape.offset(slice, 0, 0, 0);
    for (int plane(0); plane < other.shape.c; ++plane) {
      copy_gpu_to_gpu(from, to, shape.w * shape.h);      
      from += shape.w * shape.h;
      to += shape.w * shape.h;
    }
  }
}

int Volume::size() {
	return shape.size();
}

Volume &operator-=(Volume &in, Volume &other) {
	assert(in.size() == other.size());
	add_cuda<F>(other.buf->data, in.buf->data, in.size(), -1);
	return in;
}

float Volume::norm() {
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), buf->data, 1, buf->data, 1, &result) );
	return sqrt(result / size());
}

float Volume::norm2() {
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), buf->data, 1, buf->data, 1, &result) );
	return result;
}

std::vector<F> Volume::to_vector() {
	return buf->to_vector();
}

void Volume::from_vector(std::vector<F> &vec) {
	buf->from_vector(vec);
}

void Volume::thresholding(std::vector<F> &data, float threshold) {
	for (auto &v : data)
		if (v > threshold)	v = 1.0;
		else				v = 0.0;
}

void Volume::draw_slice(string filename, int slice, int channel) {
	vector<F> data = to_vector();
	//if (th > 0.0)	thresholding(data, th);

	//cout << "drawing slice " << slice << " : " << shape << " to " << filename << endl;
	write_img1c(filename, shape.w, shape.h, &data[slice * slice_size + channel * shape.w * shape.h]);
}

void Volume::draw_volume(string filename, int channel) {
  std::string name(200, ' ');
  for (size_t z(0); z < shape.z; ++z) {
    sprintf(&name[0], filename.c_str(), z);
    draw_slice(name, z, channel);
  }
}


void Volume::draw_slice_rgb(string filename, int slice) {
	vector<F> data = to_vector();
	vector<F> rgb(shape.w*shape.h*3);

	for(int c(0); c < 3; c++)
		for(int i(0); i < shape.w*shape.h; i++){
			rgb[i*3+c] = data[slice * slice_size + c * shape.w*shape.h + i];
		}
	cout << "drawing slice " << slice << " : " << shape << " to " << filename << endl;
	write_img(filename, 3, shape.w, shape.h, &rgb[0]);
}

void Volume::dropout(float p) {
	buf->rand_zero(p);
}

float *Volume::data() {
	return buf->data;
}

Volume join_volumes(vector<Volume*> volumes) {
  int total_c(0);
  assert(volumes.size() > 0);
  VolumeShape first_shape(volumes[0]->shape);
  
  for (auto v : volumes) {
    assert(v->shape.z == first_shape.z);
    assert(v->shape.w == first_shape.w);
    assert(v->shape.h == first_shape.h);
    total_c += v->shape.c;
  }

  Volume joined(VolumeShape{first_shape.z, total_c, first_shape.w, first_shape.h});
  int c(0);
  for (auto v : volumes) {
    joined.insert_volume_at_c(*v, c);
    c += v->shape.c;
  }
  return joined;
}

int VolumeShape::size() const {
	return z * c * w * h;
}

int VolumeShape::offset(int zz, int cc, int x, int y) const {
	return zz * c * w * h + cc * w * h + y * w + x;
}

int VolumeShape::offsetrm(int zz, int cc, int x, int y) const {
	return zz * c * w * h + cc * w * h + y * w + x;
}

int VolumeShape::slice_size() const {
  return c * w * h;
}

std::ostream &operator<<(std::ostream &out, VolumeShape shape) {
	return out << "[z:" << shape.z << " c:" << shape.c << " w:" << shape.w << " h:" << shape.h << "]";
}

VolumeSet::VolumeSet(VolumeShape shape_) : x(shape_), diff(shape_), shape(shape_)
{}

VolumeSet::VolumeSet(VolumeShape shape_, VolumeSet &reuse_buffer) : x(shape_, reuse_buffer.x), diff(shape_, reuse_buffer.diff), shape(shape_)
{}

void VolumeSet::zero() {
	x.zero();
	diff.zero();
}


// Volume6DSet::Volume6DSet(VolumeShape shape_) : shape(shape_) {
// 	VolumeShape &s(shape);

// 	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));
// 	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));

// 	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));
// 	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));

// 	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));
// 	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));

// 	for (auto &v : volumes) {
// 		x.push_back(&(v->x));
// 		diff.push_back(&(v->diff));
// 	}
// }

// void Volume6DSet::zero() {
// 	for (auto &v : volumes)
// 		v->zero();
// }
