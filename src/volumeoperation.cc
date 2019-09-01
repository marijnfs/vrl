#include "volumeoperation.h"
#include "vlstm.h"
#include "global.h"

using namespace std;

FCVolumeOperation::FCVolumeOperation(VolumeShape shape_, int in_map, int out_map, float dropout_) :
	op(in_map, out_map, 1, 1),
	c(out_map),
	shape(shape_),
	tin(shape_.z, in_map, shape_.w, shape_.h, 0),
	tout(shape_.z, out_map, shape_.w, shape_.h, 0),
	tin_err(shape_.z, in_map, shape_.w, shape_.h, 0),
	tout_err(shape_.z, out_map, shape_.w, shape_.h, 0),
	dropout(dropout_)
{}

void FCVolumeOperation::forward(Volume &in, Volume &out)  {
	tin.data = in.data();
	tout.data = out.data();
	op.forward(tin, tout);
	if(dropout > 0.0) {
	  if (Global::validation())
	    (*out.buf) *= 1.0 - dropout;
	  else
	    out.dropout(dropout);
	}
}

void FCVolumeOperation::backward_weights(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data();
	tout_err.data = out.diff.data();

	op.backward_weights(tin, tout_err);
	// op.scale_grad(1.0 / (shape.z * shape.w * shape.h));
}

void FCVolumeOperation::backward(VolumeSet &in, VolumeSet &out) {
	tin.data = in.x.data();
	tout.data = out.x.data();
	tin_err.data = in.diff.data();
	tout_err.data = out.diff.data();

	op.backward(tin, tout, tout_err, tin_err);
}

void FCVolumeOperation::forward_dry_run(Volume &in, Volume &out) {
	tin.data = in.data();
	tout.data = out.data();
	op.forward_dry_run(tin, tout);
}

void FCVolumeOperation::update(float lr) {
	op.update(lr);
}

void FCVolumeOperation::init_normal(float mean, float std) {
	op.init_normal(mean, std);
}

void FCVolumeOperation::init_uniform(float std) {
	op.init_uniform(std);
}

void FCVolumeOperation::register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F>> &fast_grads) {
	op.register_params(params, fast_params, grads, fast_grads);
}

VolumeShape FCVolumeOperation::output_shape(VolumeShape s) {
	return VolumeShape{s.z, c, s.w, s.h};
}

///Classify
ClassifyVolumeOperation::ClassifyVolumeOperation(VolumeShape shape_, int n_classes_) :
  n_classes(n_classes_),
  op(shape_.size(), n_classes_ , 1, 1),
  shape(shape_),
  tin(1, shape_.size(), 1, 1, 0),
  tout(1, n_classes_, 1, 1, 0),
  tin_err(1, shape_.size(), 1, 1, 0),
  tout_err(1, n_classes_, 1, 1, 0)
{}

void ClassifyVolumeOperation::forward(Volume &in, Volume &out)  {
	tin.data = in.data();
	tout.data = out.data();
	op.forward(tin, tout);
}

void ClassifyVolumeOperation::backward_weights(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data();
	tout_err.data = out.diff.data();

	op.backward_weights(tin, tout_err);
	// op.scale_grad(1.0 / (shape.z * shape.w * shape.h));
}

void ClassifyVolumeOperation::backward(VolumeSet &in, VolumeSet &out) {
	tin.data = in.x.data();
	tout.data = out.x.data();
	tin_err.data = in.diff.data();
	tout_err.data = out.diff.data();

	op.backward(tin, tout, tout_err, tin_err);
}

void ClassifyVolumeOperation::forward_dry_run(Volume &in, Volume &out) {
	tin.data = in.data();
	tout.data = out.data();
	op.forward_dry_run(tin, tout);
}

void ClassifyVolumeOperation::update(float lr) {
	op.update(lr);
}

void ClassifyVolumeOperation::init_normal(float mean, float std) {
	op.init_normal(mean, std);
}

void ClassifyVolumeOperation::init_uniform(float std) {
	op.init_uniform(std);
}

void ClassifyVolumeOperation::register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F>> &fast_grads) {
	op.register_params(params, fast_params, grads, fast_grads);
}

VolumeShape ClassifyVolumeOperation::output_shape(VolumeShape s) {
	return VolumeShape{1, n_classes, 1, 1};
}


///// Softmax
SoftmaxVolumeOperation::SoftmaxVolumeOperation(VolumeShape shape) :
	tin(shape.z, shape.c, shape.w, shape.h, 0),
	tout(shape.z, shape.c, shape.w, shape.h, 0),
	tin_err(shape.z, shape.c, shape.w, shape.h, 0),
	tout_err(shape.z, shape.c, shape.w, shape.h, 0)
{}

void SoftmaxVolumeOperation::forward(Volume &in, Volume &out){
	tin.data = in.data();
	tout.data = out.data();
	op.forward(tin, tout);
}

void SoftmaxVolumeOperation::backward(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data();
	tout.data = out.x.data();
	tin_err.data = in.diff.data();
	tout_err.data = out.diff.data();

	op.backward(tin, tout, tout_err, tin_err);
}

//Tanh Operation
TanhVolumeOperation::TanhVolumeOperation(VolumeShape shape) :
	tin(shape.z, shape.c, shape.w, shape.h, 0),
	tout(shape.z, shape.c, shape.w, shape.h, 0),
	tin_err(shape.z, shape.c, shape.w, shape.h, 0),
	tout_err(shape.z, shape.c, shape.w, shape.h, 0)
{}

void TanhVolumeOperation::forward(Volume &in, Volume &out){
	tin.data = in.data();
	tout.data = out.data();
	cout << in.data() << " " << out.data() << endl;
	cout << tin.shape() << " " << tout.shape() << endl;
	op.forward(tin, tout);
}

void TanhVolumeOperation::backward(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data();
	tout.data = out.x.data();
	tin_err.data = in.diff.data();
	tout_err.data = out.diff.data();

	op.backward(tin, tout, tout_err, tin_err);
}


//Sigmoid Operation
SigmoidVolumeOperation::SigmoidVolumeOperation(VolumeShape shape) :
	tin(shape.z, shape.c, shape.w, shape.h, 0),
	tout(shape.z, shape.c, shape.w, shape.h, 0),
	tin_err(shape.z, shape.c, shape.w, shape.h, 0),
	tout_err(shape.z, shape.c, shape.w, shape.h, 0)
{}

void SigmoidVolumeOperation::forward(Volume &in, Volume &out){
	tin.data = in.data();
	tout.data = out.data();
	op.forward(tin, tout);
}

void SigmoidVolumeOperation::backward(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data();
	tout.data = out.x.data();
	tin_err.data = in.diff.data();
	tout_err.data = out.diff.data();

	op.backward(tin, tout, tout_err, tin_err);
}

//Volume Operation
TimeOperation1::TimeOperation1(Operation<F> &op_, VolumeSet &in_, VolumeSet &out_, int dt_, float beta_) :
	op(op_), T(in_.x.shape.z), dt(dt_), beta(beta_),
	in(in_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
{}

void TimeOperation1::forward(int t) {
	if (t < dt)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);
	// cout << "slice shape " << in.x.slice_shape() << " " << out.x.slice_shape() << endl;
	op.forward(in_t, out_t, beta);
}

void TimeOperation1::backward(int t) {
	if (t < dt)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	in_err_t.data = in.diff.slice(t - dt);
	out_err_t.data = out.diff.slice(t);

	op.backward(in_t, out_t, out_err_t, in_err_t, beta);
	op.backward_weights(in_t, out_err_t, 1.0);
}

void TimeOperation1::forward_dry_run() {
	// cout << in_t.shape() << " " << out_t.shape() << endl;
	op.forward_dry_run(in_t, out_t);
}


////////////
//Volume Operation Rolled out
TimeOperation1Rollout::TimeOperation1Rollout(Operation<F> &op_, VolumeSet &in_, VolumeSet &out_, int dt_, float beta_) :
	op(op_), T(in_.x.shape.z), dt(dt_), beta(beta_),
	in(in_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
{}

void TimeOperation1Rollout::forward(int t) {
	if (t < dt)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	// cout << "forward times " << t << endl;
	op.forward_timed(in_t, out_t, t, beta); //for fastweight purposes
}

void TimeOperation1Rollout::backward(int t) {
	if (t < dt)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	in_err_t.data = in.diff.slice(t - dt);
	out_err_t.data = out.diff.slice(t);

	op.backward_timed(in_t, out_t, out_err_t, in_err_t, t, beta);
	op.backward_weights_timed(in_t, out_err_t, t, 1.0);
}

void TimeOperation1Rollout::forward_dry_run() {
	// cout << in_t.shape() << " " << out_t.shape() << endl;
	op.forward_dry_run(in_t, out_t);
}


////////
//Time operation 2 arguments (for gating)
TimeOperation2::TimeOperation2(Operation2<F> &op_, VolumeSet &in_, VolumeSet &in2_, VolumeSet &out_, int dt_, float beta_) :
	op(op_),T(in_.x.shape.z), dt(dt_), beta(beta_), in(in_), in2(in2_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	in2_t(in2_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	in2_err_t(in2_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
{}

void TimeOperation2::forward(int t) {
	if (t < dt)
		return;
	//we only delay the first input
	in_t.data = in.x.slice(t - dt);
	in2_t.data = in2.x.slice(t);
	out_t.data = out.x.slice(t);

	op.forward(in_t, in2_t, out_t, beta);
}

void TimeOperation2::backward(int t) {
	if (t < dt)
		return;
	in_t.data = in.x.slice(t - dt);
	in2_t.data = in2.x.slice(t);
	out_t.data = out.x.slice(t);

	in_err_t.data = in.diff.slice(t - dt);
	in2_err_t.data = in2.diff.slice(t);
	out_err_t.data = out.diff.slice(t);

	op.backward(in_t, in2_t, out_t, out_err_t, in_err_t, in2_err_t, beta);
	//op.backward_weights(in_t, );
}

void TimeOperation2::forward_dry_run() {
	op.forward_dry_run(in_t, in2_t, out_t);
}

VolumeShape output_shape(VolumeShape in, Operation<F> &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h};
}

VolumeShape output_shape(VolumeShape in, Operation2<F> &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h};
}
