#include "volumenetwork.h"
#include "volumeoperation.h"
#include "vlstm.h"

using namespace std;

VolumeNetwork::VolumeNetwork(VolumeShape shape) : n_params(0) {
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::forward() {
	clear();

	for (int i(0); i < operations.size(); i++) {
	  //cout << "===" << i << endl;
		handle_error( cudaGetLastError() );

		operations[i]->forward(volumes[i]->x, volumes[i+1]->x);
		handle_error( cudaGetLastError() );
	}
}

void VolumeNetwork::backward() {
	for (int i(operations.size() - 1); i >= 0; i--) {
		operations[i]->backward(*volumes[i], *volumes[i+1]);
		operations[i]->backward_weights(*volumes[i], *volumes[i+1]);
	}
	// grad_vec *= output_shape().size(); //loss is counter for every pixel, normalise
}

void VolumeNetwork::forward_dry_run() {
	for (int i(0); i < operations.size(); i++) {
		operations[i]->forward_dry_run(volumes[i]->x, volumes[i+1]->x);
	}
}

void VolumeNetwork::finish() {
	forward_dry_run();

	register_params();
	align_params();
	for(auto o : operations)
		o->sharing();
	clear();

	//init_normal(0, 0);
	// a.resize(param.n);
	// b.resize(param.n);
	// c.resize(param.n);
	// d.resize(param.n);
	// e.resize(param.n);
	// rmse.resize(param.n);
	// rmse += .01;
}

void VolumeNetwork::register_params() {
	for (auto &o : operations)
		o->register_params(param_ptrs, fast_param_ptrs, grad_ptrs, fast_grad_ptrs);

 	n_params = 0;
	for (auto &p : param_ptrs)
		n_params += p.n;

	n_fast_params = 0;
	for (auto &p : fast_param_ptrs)
		n_fast_params += p.n;
}

void VolumeNetwork::align_params() {
	param_vec.resize(n_params);
	grad_vec.resize(n_params);

	fast_param_vec.resize(n_fast_params);
	fast_grad_vec.resize(n_fast_params);

	for (auto &p : param_ptrs)
		cudaFree(*(p.ptr));
	for (auto &p : fast_param_ptrs)
		cudaFree(*(p.ptr));

	for (auto &g : grad_ptrs)
		cudaFree(*(g.ptr));
	for (auto &g : fast_grad_ptrs)
		cudaFree(*(g.ptr));

	position_params(param_vec.data, fast_param_vec.data, grad_vec.data, fast_grad_vec.data);
	cout << "n params: " << n_params << endl;
	//throw "";
}

void VolumeNetwork::position_params(float *pos_param, float *pos_fast_param, float *pos_grad, float *pos_fast_grad) {
	float *ptr = pos_param;
	for (auto &p : param_ptrs) {
		*(p.ptr) = ptr;
		ptr += p.n;
	}

	ptr = pos_fast_param;
	for (auto &p : fast_param_ptrs) {
		*(p.ptr) = ptr;
		ptr += p.n;
	}

	ptr = pos_grad;
	for (auto &g : grad_ptrs) {
		*(g.ptr) = ptr;
		ptr += g.n;
	}

	ptr = pos_fast_grad;
	for (auto &g : fast_grad_ptrs) {
		*(g.ptr) = ptr;
		ptr += g.n;
	}
}

void VolumeNetwork::update(float lr) {
	for (int i(0); i < operations.size(); i++)
		operations[i]->update(lr);
}

void VolumeNetwork::clear() {
	for (int i(0); i < volumes.size(); i++) {
		// if (!i)
		// 	volumes[i]->diff.zero();
		// else
		// 	volumes[i]->zero();
		if (i)
			volumes[i]->x.zero();
		volumes[i]->diff.zero();
	}
}

void VolumeNetwork::init_normal(float mean, float std) {
	for (auto &o : operations)
		o->init_normal(mean, std);
}

void VolumeNetwork::init_uniform(float var) {
	for (auto &o : operations)
		o->init_uniform(var);
}

void VolumeNetwork::set_input(Volume &in) {
	first(volumes)->x.from_volume(in);
}

Volume &VolumeNetwork::output() {
	return last(volumes)->x;
}

Volume &VolumeNetwork::input() {
	return first(volumes)->x;
}

float VolumeNetwork::calculate_loss(Volume &target) {
	last(volumes)->diff.from_volume(target);
	last(volumes)->diff -= last(volumes)->x;
	// last(volumes)->diff.from_volume(last(volumes)->x);
	// last(volumes)->diff -= target;

	float norm = last(volumes)->diff.norm2() * 0.5;
	//float norm = last(volumes)->diff.norm();
	//return norm;
	return norm;
}

void VolumeNetwork::add_vlstm(int kg, int ko, int c) {
	cout << "adding vlstm" << endl;
	//cout << "adding: " << last(shapes) << " " << shape << endl;

	auto vlstm = new VLSTMOperation(last(shapes), kg, ko, c, vsm);
	auto shape = vlstm->output_shape(last(shapes));

	operations.push_back(vlstm);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_hwv(int kw) {
	cout << "adding hwv" << endl;
	//cout << "adding: " << last(shapes) << " " << shape << endl;

	auto hwv = new HWVOperation(last(shapes), kw, vsm);
	auto shape = hwv->output_shape(last(shapes));

	operations.push_back(hwv);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_univlstm(int kg, int ko, int c) {
	cout << "adding unidirectional vlstm" << endl;
	//cout << "adding: " << last(shapes) << " " << shape << endl;

	auto vlstm = new UniVLSTMOperation(last(shapes), kg, ko, c, vsm);
	auto shape = vlstm->output_shape(last(shapes));

	operations.push_back(vlstm);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_fc(int c, float dropout) {
	cout << "adding fc" << endl;
	auto fc = new FCVolumeOperation(last(shapes), last(shapes).c, c, dropout);
	auto shape = fc->output_shape(last(shapes));
	operations.push_back(fc);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_classify(int n_classes) {
	cout << "adding classification" << endl;
	auto fc = new ClassifyVolumeOperation(last(shapes), n_classes);
	auto shape = fc->output_shape(last(shapes));
	operations.push_back(fc);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_softmax() {
	cout << "adding softmax" << endl;
	auto softmax = new SoftmaxVolumeOperation(last(shapes));
	auto shape = softmax->output_shape(last(shapes));
	operations.push_back(softmax);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_tanh() {
	cout << "adding tanh" << endl;
	auto tan_op = new TanhVolumeOperation(last(shapes));
	auto shape = tan_op->output_shape(last(shapes));
	operations.push_back(tan_op);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_sigmoid() {
	cout << "adding sigmoid" << endl;
	auto sig = new SigmoidVolumeOperation(last(shapes));
	auto shape = sig->output_shape(last(shapes));
	operations.push_back(sig);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::save(std::string path) {
	ofstream of(path, ios::binary);
	vector<float> data = param_vec.to_vector();
	byte_write_vec(of, data);
}

void VolumeNetwork::load(std::string path) {
	ifstream in(path, ios::binary);
	vector<float> data = byte_read_vec<float>(in);
	if (data.size() != param_vec.n) {
		cout << data.size() << " != " << param_vec.n << endl;
		throw StringException("load size does not match");
	}
	param_vec.from_vector(data);
}

void VolumeNetwork::describe(std::ostream &out) {
	for (auto &o : operations) {
		o->describe(out);
		out << endl;
	}
	out.flush();
}

void VolumeNetwork::set_fast_weights(Tensor<float> &weights) {
	//weights come out of network ordered weights first and then by time, while they are packed reversely
	//we account for that here
	int T = input().shape.z;

	int shift(0);
	for (CudaPtr<F> &param : fast_param_ptrs) {
		int n = param.n / T;
		for (size_t t(0); t < T; ++t) {
			F *dest = (*param.ptr) + t * n;
			F *src = weights.ptr() + t * weights.c + shift;
			copy_gpu_to_gpu<>(src, dest, n);
		}
		shift += n;
	}
}

void VolumeNetwork::get_fast_grads(Tensor<float> &grad_tensor) {
	//weights come out of network ordered weights first and then by time, while they are packed reversely
	//we account for that here
	int T = input().shape.z;

	int shift(0);
	for (CudaPtr<F> &grad : fast_grad_ptrs) {
		int n = grad.n / T;
		// cout << "get grads " << grad.n << " " << T << " " << n << endl;
		// vector<float> bla(n);
		for (size_t t(0); t < T; ++t) {
			F *src = (*grad.ptr) + t * n;
			F *dest = grad_tensor.ptr() + t * grad_tensor.c + shift;
			// cout << (fast_grad_vec.data) << " " << src << endl;
			// copy_gpu_to_cpu<>(src, &bla[0], n);
			// cout << bla << endl;
			copy_gpu_to_gpu<>(src, dest, n);
		}
		shift += n;
	}
}
