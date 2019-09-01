#include "vlstm.h"
#include "divide.h"
#include <typeinfo>
#include <iostream>

using namespace std;

SubVolumeOperation::SubVolumeOperation(VolumeShape in) :
in_shape(in),
vin(0),	vout(0),
T(in.z),
n_param(0)
{}



void SubVolumeOperation::update(float lr) {
	for (auto &p : parameters) {
		p->update(lr);
		//cout << p->grad_to_vector() << endl;
	}

}

void SubVolumeOperation::clear() {
	for (auto& v : volumes) {
		 // if (v.first != "x" && v.first != "h")
	 	v.second->x.zero();
		//v.second->x.zero();
		v.second->diff.zero();
	}
}

void SubVolumeOperation::clear_grad() {
	for (auto& p : parameters)
		p->zero_grad();
}
void SubVolumeOperation::scale_grad() {
	for (auto &p : parameters)
		p->scale_grad(1.0);
}


void SubVolumeOperation::add_volume(string name, VolumeShape shape, VolumeSetMap *reuse) {
	if (reuse) {
		if (!(*reuse).count(name))
			(*reuse)[name] = new VolumeSet(shape);
		volumes[name] = new VolumeSet(shape, *(*reuse)[name]);
	}
	else
		volumes[name] = new VolumeSet(shape);
}

void SubVolumeOperation::add_volume(string name, VolumeSet &set) {
	volumes[name] = &set;
}

bool SubVolumeOperation::exists(string name) {
	return volumes.count(name);
}

void SubVolumeOperation::init_normal(F mean, F std) {
	for (auto &p : parameters)
		p->init_normal(mean, std);
}

void SubVolumeOperation::init_uniform(F var) {
	for (auto &p : parameters)
		p->init_uniform(var);
}

void SubVolumeOperation::register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F>> &fast_grads) {
	for (auto &p : parameters)
		p->register_params(params, fast_params, grads, fast_grads);
}

void SubVolumeOperation::forward_dry_run() {
	for (auto &op : operations)
		op->forward_dry_run();
}

void SubVolumeOperation::forward() {
//	int T(in_shape.z);
	for (int t(0); t < T; ++t) {
		// cout << "forward t:" << t << endl;
		for (auto &op : operations) {
			op->forward(t);
		}
	}
}

void SubVolumeOperation::backward() {
	// cout << "back" << endl;
	for (int t(T - 1); t >= 0; --t)
		for (int n(operations.size() - 1); n >= 0; --n) {
			operations[n]->backward(t);
		}
	// // cout << "scaling" << endl;
	// for (auto &p : parameters)
	// 	//p->scale_grad(1.0 / (in_shape.z * in_shape.w * in_shape.h));
	// 	p->scale_grad(1.0 / sqrt(in_shape.z * in_shape.w * in_shape.h));
	// cout << "done" << endl;
}

void SubVolumeOperation::add_op(string ins, string outs, Operation<F> &op, bool delay, VolumeSetMap *reuse, float beta, bool param) {
	VolumeSet &in(*volumes[ins]);

	if (!exists(outs))
		add_volume(outs, output_shape(in.shape, op), reuse);

	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;


	operations.push_back(new TimeOperation1(op, in, out, dt, beta));
	if (param) {
		try {
		  Parametrised<F> &p = dynamic_cast<Parametrised<F> &>(op);
		  parameters.push_back(&p);
		  cout << ins << "-" << outs << ":" << n_param << "-" << (n_param + p.size()) << endl;
		  n_param += p.size();
			// cout << "a parameter" << endl;
		} catch (const std::bad_cast& e) {
			// cout << "not a parameter" << endl;
		}
	}
}


void SubVolumeOperation::add_op_rollout(string ins, string outs, Operation<F> &op, bool delay, VolumeSetMap *reuse, float beta) {
	VolumeSet &in(*volumes[ins]);

	if (!exists(outs))
		add_volume(outs, output_shape(in.shape, op), reuse);

	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;

	operations.push_back(new TimeOperation1Rollout(op, in, out, dt, beta));
	try {
		parameters.push_back(&dynamic_cast<Parametrised<F> &>(op));
		// cout << "a parameter" << endl;
	} catch (const std::bad_cast& e) {
		// cout << "not a parameter" << endl;
	}
}

void SubVolumeOperation::add_op(string ins, string in2s, string outs, Operation2<F> &op, bool delay, VolumeSetMap *reuse, float beta) {
	VolumeSet &in(*volumes[ins]);
	VolumeSet &in2(*volumes[in2s]);

	if (!exists(outs))
		add_volume(outs, output_shape(in.shape, op), reuse);

	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;
	operations.push_back(new TimeOperation2(op, in, in2, out, dt, beta));
}



//LSTM operation
LSTMOperation::LSTMOperation(VolumeShape in, int kg, int ko, int c, VolumeSetMap *reuse) :
	SubVolumeOperation(in),
	xi(in.c, c, kg, kg), hi(c, c, kg, kg), //input gate
	xr(in.c, c, kg, kg), hr(c, c, kg, kg), //remember gate (forget gates dont make sense!)
	xs(in.c, c, kg, kg), hs(c, c, kg, kg), //cell input
	xo(in.c, c, ko, ko), ho(c, c, ko, ko), //co(c, c, ko, ko), //output gate
	cc(c, c, 3, 3)
{
	add_volume("x", VolumeShape{in.z, in.c, in.w, in.h}, reuse);
	add_volume("h", VolumeShape{in.z, c, in.w, in.h}, reuse);

	add_operations(reuse);

	vin = volumes["x"];
	vout = volumes["h"];

	xr.bias.init_normal(1., 0.0);
	xi.bias.init_normal(1., 0.0);
	cc.filter_bank.init_normal(1.0 / cc.filter_bank.n_weights(), 0.);
}

//LSTM operation
LSTMOperation::LSTMOperation(VolumeShape in, int kg, int ko, int c, bool rollout, VolumeSetMap *reuse) :
	SubVolumeOperation(in),
	xi("", in.c, c, kg, kg, in.z), hi("", c, c, kg, kg, in.z), //input gate
	xr("", in.c, c, kg, kg, in.z), hr("", c, c, kg, kg, in.z), //remember gate (forget gates dont make sense!)
	xs("", in.c, c, kg, kg, in.z), hs("", c, c, kg, kg, in.z), //cell input
	xo("", in.c, c, ko, ko, in.z), ho("", c, c, ko, ko, in.z), //co(c, c, ko, ko), //output gate
	cc(c, c, 3, 3)
{
	add_volume("x", VolumeShape{in.z, in.c, in.w, in.h}, reuse);
	add_volume("h", VolumeShape{in.z, c, in.w, in.h}, reuse);

	if (rollout)
		add_operations_rollout(reuse);
	else
		add_operations(reuse);

	vin = volumes["x"];
	vout = volumes["h"];

	xr.bias.init_normal(1., 0.0);
	xi.bias.init_normal(1., 0.0);
	cc.filter_bank.init_normal(1.0 / cc.filter_bank.n_weights(), 0.);

	// cc.filter_bank.init_normal(1.0 / 9.0, 0.);
	// xo.bias.init_normal(1., 0.0);
}

void LSTMOperation::add_operations(VolumeSetMap *reuse) {
	bool DELAY(true), NOW(false);

	//Start
	add_op("h", "i", hi, DELAY, reuse);
	add_op("x", "i", xi, NOW, reuse);
	add_op("i", "fi", sig, NOW, reuse);
	// add_op("i", "i", sig, NOW, reuse, 0.0);

	add_op("h", "r", hr, DELAY, reuse);
	add_op("x", "r", xr, NOW, reuse);
	add_op("r", "fr", sig, NOW, reuse);
	// add_op("r", "r", sig, NOW, reuse, 0.0);

	add_op("h", "s", hs, DELAY, reuse);
	add_op("x", "s", xs, NOW, reuse);
	add_op("s", "fs", tan, NOW, reuse);
	// add_op("s", "s", tan, NOW, reuse, 0.0);

	add_op("fs", "fi", "c", gate, NOW, reuse);
	// add_op("c", "ct", cc, DELAY, reuse, 0.0, false); //hack
	// add_op("ct", "fr", "c", gate, NOW, reuse); //hack
	add_op("c", "fr", "c", gate, DELAY, reuse); //

	add_op("c", "fc", tan, NOW, reuse);

	add_op("x", "o", xo, NOW, reuse);
	add_op("h", "o", ho, DELAY, reuse);
	add_op("o", "fo", sig, NOW, reuse);
	// add_op("o", "o", sig, NOW, reuse, 0.0);

	add_op("fc", "fo", "h", gate, NOW, reuse);

	//Direct conv
	//add_op("x", "h", xo, DELAY, reuse);

}


void LSTMOperation::add_operations_rollout(VolumeSetMap *reuse) {
	bool DELAY(true), NOW(false);

	//Start
	add_op_rollout("x", "i", xi, NOW, reuse);
	add_op_rollout("h", "i", hi, DELAY, reuse);
	add_op("i", "fi", sig, NOW, reuse);

	add_op_rollout("x", "r", xr, NOW, reuse);
	add_op_rollout("h", "r", hr, DELAY, reuse);
	add_op("r", "fr", sig, NOW, reuse);

	add_op_rollout("x", "s", xs, NOW, reuse);
	add_op_rollout("h", "s", hs, DELAY, reuse);
	add_op("s", "fs", tan, NOW, reuse);

	add_op("fs", "fi", "c", gate, NOW, reuse);
	// add_op("c", "ct", cc, DELAY, reuse, 0.0, false); //hack
	// add_op("ct", "fr", "c", gate, NOW, reuse); //hack
	add_op("c", "fr", "c", gate, DELAY, reuse); //

	add_op("c", "fc", tan, NOW, reuse);

	add_op_rollout("x", "o", xo, NOW, reuse);
	add_op_rollout("h", "o", ho, DELAY, reuse);
	//add_op("c", "o", co, reuse);
	add_op("o", "fo", sig, NOW, reuse);

	add_op("fc", "fo", "h", gate, NOW, reuse);

	 // add_op("x", "h", xo, NOW, reuse);
}


void LSTMOperation::share(LSTMOperation &o){
	xi.share(o.xi);
	hi.share(o.hi); //input gate
	xr.share(o.xr);
	hr.share(o.hr); //remember gate (forget gates dont make sense!)
	xs.share(o.xs);
	hs.share(o.hs); //cell input
	xo.share(o.xo);
	ho.share(o.ho);
	// co.share(o.co); //output gate
}


LSTMShiftOperation::LSTMShiftOperation(VolumeShape in, int kg, int ko, int c, int dx, int dy, VolumeSetMap *reuse) :
	SubVolumeOperation(in),
	xi(in.c, c, kg, kg, dx, dy), hi(c, c, kg, kg, dx, dy), //input gate
	xr(in.c, c, kg, kg, dx, dy), hr(c, c, kg, kg, dx, dy), //remember gate (forget gates dont make sense!)
	xs(in.c, c, kg, kg, dx, dy), hs(c, c, kg, kg, dx, dy), //cell input
	xo(in.c, c, ko, ko, dx, dy), ho(c, c, ko, ko, dx, dy), //co(c, c, ko, ko), //output gate
	cc(c, c, 3, 3)
{
	add_volume("x", VolumeShape{in.z, in.c, in.w, in.h}, reuse);
	add_volume("h", VolumeShape{in.z, c, in.w, in.h}, reuse);

	add_operations(reuse);

	vin = volumes["x"];
	vout = volumes["h"];

	// xr.bias.init_normal(1., 0.0);
	// xi.bias.init_normal(1., 0.0);
}


void LSTMShiftOperation::add_operations(VolumeSetMap *reuse) {
	bool DELAY(true), NOW(false);
	// add_op("x", "i", xi, NOW, reuse);
	// add_op("h", "i", hi, DELAY, reuse);
	// add_op("i", "fi", sig, NOW, reuse);

	// add_op("x", "r", xr, NOW, reuse);
	// add_op("h", "r", hr, DELAY, reuse);
	// add_op("r", "fr", sig, NOW, reuse);

	// add_op("x", "s", xs, NOW, reuse);
	// add_op("h", "s", hs, DELAY, reuse);
	// add_op("s", "fs", tan, NOW, reuse);

	// add_op("fs", "fi", "c", gate, NOW, reuse);
	// add_op("c", "fr", "c", gate, DELAY, reuse);
	// // add_op("c", "fc", tan, reuse);
	// add_op("c", "fc", sig, NOW, reuse);

	// add_op("x", "o", xo, NOW, reuse);
	// add_op("h", "o", ho, DELAY, reuse);
	// //add_op("c", "o", co, reuse);
	// add_op("o", "fo", sig, NOW, reuse);

	// add_op("fc", "fo", "h", gate, NOW, reuse);

	//Start
	add_op("x", "i", xi, NOW, reuse);
	add_op("h", "i", hi, DELAY, reuse);
	add_op("i", "fi", sig, NOW, reuse, 0.0);

	add_op("x", "r", xr, NOW, reuse);
	add_op("h", "r", hr, DELAY, reuse);
	add_op("r", "fr", sig, NOW, reuse, 0.0);

	add_op("x", "s", xs, NOW, reuse);
	add_op("h", "s", hs, DELAY, reuse);
	add_op("s", "fs", tan, NOW, reuse, 0.0);

	add_op("fs", "fi", "c", gate, NOW, reuse);
	// add_op("c", "ct", cc, DELAY, reuse); //hack
	// add_op("ct", "fr", "c", gate, NOW, reuse); //hack
	add_op("c", "fr", "c", gate, DELAY, reuse); //

	add_op("c", "fc", tan, NOW, reuse);

	add_op("x", "o", xo, NOW, reuse);
	add_op("h", "o", ho, DELAY, reuse);
	//add_op("c", "o", co, reuse);
	add_op("o", "fo", sig, NOW, reuse, 0.0);

	add_op("fc", "fo", "h", gate, NOW, reuse);

	 // add_op("x", "h", xo, NOW, reuse);
}

HWOperation::HWOperation(VolumeShape in, int kw, VolumeSetMap *reuse) :
  SubVolumeOperation(in),
  hg(in.c, in.c, kw, kw),
  hh(in.c, in.c, kw, kw),
  xg(in.c, in.c, kw, kw),
  xh(in.c, in.c, kw, kw)
{
  add_volume("x", VolumeShape{in.z, in.c, in.w, in.h}, reuse);
  add_volume("h", VolumeShape{in.z, in.c, in.w, in.h}, reuse);

  add_hw_operations(reuse);

  vin = volumes["x"];
  vout = volumes["h"];

  //xr.bias.init_normal(1., 0.0);
  //xi.bias.init_normal(1., 0.0);
  //cc.filter_bank.init_normal(1.0 / cc.filter_bank.n_weights(), 0.);
}

//~HWOperation::HWOperation(){}

void HWOperation::add_hw_operations(VolumeSetMap *reuse) {
  bool DELAY(true), NOW(false);
  //get gates
  add_op("h", "gl", hg, DELAY, reuse);
  add_op("gl", "g", sig, NOW, reuse);

  add_op("h", "g", "h", gate, DELAY, reuse);
  
  //get hidden input
  add_op("h", "hhl", hh, DELAY, reuse);
  add_op("hhl", "h", tan, NOW, reuse);

  //gate hidden input in hidden
  
  //add input
  add_op("x", "xl", xh, NOW, reuse);
  add_op("xl", "h", tan, NOW, reuse);
  
  add_op("x", "xgl", xg, NOW, reuse);
  add_op("xgl", "xg", sig, NOW, reuse);
  add_op("x", "xg", "h", gate, NOW, reuse);
}
    
void HWOperation::share(HWOperation &o) {
  hg.share(o.hg);
  hh.share(o.hh);
  xg.share(o.xg);
  xh.share(o.xh);
}




VLSTMOperation::VLSTMOperation() : kg(0), ko(0), c(0) { //mostly for overloading
}

//Vlstm
VLSTMOperation::VLSTMOperation(VolumeShape s, int kg_, int ko_, int c_, VolumeSetMap &vsm):
kg(kg_), ko(ko_), c(c_)
{
	// for (size_t i(0); i < 6; ++i)
		// operations.push_back(new LSTMOperation(*(x6.volumes[i]), *(y6.volumes[i]), kg, ko, c));

	operations.push_back(new LSTMOperation(VolumeShape{s.z, s.c, s.w, s.h}, kg, ko, c, &vsm));
	operations.push_back(new LSTMOperation(VolumeShape{s.z, s.c, s.w, s.h}, kg, ko, c, &vsm));

	operations.push_back(new LSTMOperation(VolumeShape{s.w, s.c, s.z, s.h}, kg, ko, c, &vsm));
	operations.push_back(new LSTMOperation(VolumeShape{s.w, s.c, s.z, s.h}, kg, ko, c, &vsm));

	operations.push_back(new LSTMOperation(VolumeShape{s.h, s.c, s.w, s.z}, kg, ko, c, &vsm));
	operations.push_back(new LSTMOperation(VolumeShape{s.h, s.c, s.w, s.z}, kg, ko, c, &vsm));


	prepare();
}

void VLSTMOperation::prepare() {
  clear();
  for (auto &op : operations)
    op->forward_dry_run();
}

void VLSTMOperation::sharing() {
	///sharing weights
	// operations[0]->share(*operations[1]);
	// operations[2]->share(*operations[3]);
	// operations[2]->share(*operations[4]);
	// operations[2]->share(*operations[5]);
}

void VLSTMOperation::clear() {
	for (auto& o : operations) {
		o->clear();
		o->clear_grad();
	}
}

void VLSTMOperation::forward(Volume &in, Volume &out) {
	for (size_t i(0); i < operations.size(); ++i) {
		operations[i]->clear();
		divide(in, operations[i]->input().x, i);

		operations[i]->forward();
		combine(operations[i]->output().x, out, i);
		// if (i == 0) {
		// 	operations[0]->volumes["c"]->x.draw_slice("c1.png", 1);
		// 	operations[0]->volumes["c"]->x.draw_slice("c3.png", 3);
		// 	operations[0]->volumes["i"]->x.draw_slice("i1.png", 1);
		// 	operations[0]->volumes["i"]->x.draw_slice("i3.png", 3);
		// 	operations[0]->volumes["o"]->x.draw_slice("o1.png", 1);
		// 	operations[0]->volumes["o"]->x.draw_slice("o3.png", 3);
		// }
	}
}

void VLSTMOperation::backward(VolumeSet &in, VolumeSet &out) {
	for (size_t i(0); i < operations.size(); ++i) {
		//forward
		operations[i]->clear_grad();
		operations[i]->clear();
		divide(in.x, operations[i]->input().x, i);
		operations[i]->forward();
		// if (i == 2) {
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[0])->in.x.draw_slice("inx0.png",0);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[0])->out.x.draw_slice("outi0.png",0);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[12])->out.x.draw_slice("outcf0.png",0);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[0])->in.x.draw_slice("inx5.png",5);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[0])->out.x.draw_slice("outi5.png",5);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[12])->out.x.draw_slice("outcf5.png",5);

		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[0])->in.x.draw_slice("inx10.png",10);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[0])->out.x.draw_slice("outi10.png",10);
		// 	dynamic_cast<TimeOperation1*>(operations[i]->operations[12])->out.x.draw_slice("outcf10.png",10);
		// }


		//backward
		divide(out.diff, operations[i]->output().diff, i);
		operations[i]->backward();
		combine(operations[i]->input().diff, in.diff, i);
		// operations[i]->scale_grad();
	}
}

void VLSTMOperation::forward_dry_run(Volume &in, Volume &out){

}

VolumeShape VLSTMOperation::output_shape(VolumeShape s) {
	return VolumeShape{s.z, c, s.w, s.h};
}


void VLSTMOperation::init_normal(F mean, F std) {
	for (auto &o : operations)
		o->init_normal(mean, std);
}

void VLSTMOperation::init_uniform(F std) {
	for (auto &o : operations)
		o->init_uniform(std);
}

void VLSTMOperation::register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F>> &fast_grads) {
	for (auto &o : operations)
		o->register_params(params, fast_params, grads, fast_grads);

	// operations[0]->register_params(params, grads);
	// operations[2]->register_params(params, grads);
}

void VLSTMOperation::update(float lr) {
	for (auto &o : operations) {
		// cout << "update lstm op" << endl;
		o->update(lr);
	}
/*	operations[0]->update(lr);
	operations[2]->update(lr);*/
}

HWVOperation::HWVOperation(VolumeShape s, int kw_, VolumeSetMap &vsm) {
  kw = kw_;
  c = s.c;
  
  operations.push_back(new HWOperation(VolumeShape{s.z, s.c, s.w, s.h}, kw, &vsm));
  operations.push_back(new HWOperation(VolumeShape{s.z, s.c, s.w, s.h}, kw, &vsm));
  
  operations.push_back(new HWOperation(VolumeShape{s.w, s.c, s.z, s.h}, kw, &vsm));
  operations.push_back(new HWOperation(VolumeShape{s.w, s.c, s.z, s.h}, kw, &vsm));
  
  operations.push_back(new HWOperation(VolumeShape{s.h, s.c, s.w, s.z}, kw, &vsm));
  operations.push_back(new HWOperation(VolumeShape{s.h, s.c, s.w, s.z}, kw, &vsm));
    
  prepare();
}

//Vlstm
UniVLSTMOperation::UniVLSTMOperation(VolumeShape s, int kg_, int ko_, int c_, VolumeSetMap &vsm)
{
	kg = kg_;
	ko = ko_;
	c = c_;

	// for (size_t i(0); i < 6; ++i)
		// operations.push_back(new LSTMOperation(*(x6.volumes[i]), *(y6.volumes[i]), kg, ko, c));

	bool rollout(true);///true
	operations.push_back(new LSTMOperation(VolumeShape{s.z, s.c, s.w, s.h}, kg, ko, c, rollout, &vsm));
	// bool rollout(false);///true
	// operations.push_back(new LSTMOperation(VolumeShape{s.z, s.c, s.w, s.h}, kg, ko, c, &vsm));


	// operations.push_back(new LSTMShiftOperation(VolumeShape{s.w, s.c, s.z, s.h}, kg, ko, c, 1, 0, &vsm)); //both +x, for the time direction is on this axis (check divide)
	// operations.push_back(new LSTMShiftOperation(VolumeShape{s.w, s.c, s.z, s.h}, kg, ko, c, 1, 0, &vsm));

	// operations.push_back(new LSTMShiftOperation(VolumeShape{s.h, s.c, s.w, s.z}, kg, ko, c, 0, 1, &vsm)); //both +y, for the time direction is on this axis (check divide)
	// operations.push_back(new LSTMShiftOperation(VolumeShape{s.h, s.c, s.w, s.z}, kg, ko, c, 0, 1, &vsm));

	clear();
	for (auto &op : operations)
		op->forward_dry_run();
}

void UniVLSTMOperation::forward(Volume &in, Volume &out) {
	vector<Direction> directions={ZF};//, XF, XB, YF, YB};
	// vector<Direction> directions={ZF};//, XF, XB};
	for (size_t i(0); i < directions.size(); ++i) {

		operations[i]->clear();
		// cout << "forward dir:" << i << " " << directions[i] << endl;
		// cout << in.shape << " " << operations[i]->input().x.shape << endl;
		divide(in, operations[i]->input().x, directions[i]);
		// in.draw_slice("middlein.png", 0);
		// in.draw_slice("middlein2.png", 10);

		// operations[i]->input().x.draw_slice("middle.png", 0);
		// operations[i]->input().x.draw_slice("middle2.png", 10);
		// operations[i]->input().x.draw_slice("middle3.png", 200);
		operations[i]->forward();
		// operations[i]->output().x.draw_slice("middleout.png", 0);
		// operations[i]->output().x.draw_slice("middleout2.png", 10);
		// operations[i]->output().x.draw_slice("middleout3.png", 200);

		// combine(operations[i]->output().x, out, directions[i]);
		combine(operations[i]->output().x, out, directions[i]);
		// if (i == 0) {
		// 	operations[0]->volumes["c"]->x.draw_slice("c1.png", 1);
		// 	operations[0]->volumes["c"]->x.draw_slice("c3.png", 3);
		// 	operations[0]->volumes["i"]->x.draw_slice("i1.png", 1);
		// 	operations[0]->volumes["i"]->x.draw_slice("i3.png", 3);
		// 	operations[0]->volumes["o"]->x.draw_slice("o1.png", 1);
		// 	operations[0]->volumes["o"]->x.draw_slice("o3.png", 3);
		// }
	}
}

void UniVLSTMOperation::backward(VolumeSet &in, VolumeSet &out) {
	// operations[0]->clear_grad();
	// operations[2]->clear_grad();
	vector<Direction> directions={ZF};//, XF, XB, YF, YB};
	// vector<Direction> directions={ZF};//, XF, XB};
	for (size_t i(0); i < directions.size(); ++i) {
		//forward
		operations[i]->clear_grad();
		operations[i]->clear();
		divide(in.x, operations[i]->input().x, directions[i]);
		operations[i]->forward();

		//backward
		divide(out.diff, operations[i]->output().diff, directions[i]);
		operations[i]->backward();
		combine(operations[i]->input().diff, in.diff, directions[i]);
		// operations[i]->scale_grad();
	}
}
