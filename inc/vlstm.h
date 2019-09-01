#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <string>
#include <map>
#include <vector>

#include "volume.h"
#include "operations.h"
#include "volumeoperation.h"


struct SubVolumeOperation {
	SubVolumeOperation(VolumeShape in);
	virtual ~SubVolumeOperation(){}

	void add_op(std::string in, std::string out, Operation<F> &op, bool delay = false, VolumeSetMap *reuse = 0, float beta = 1.0, bool param = true);
	void add_op(std::string in, std::string in2, std::string out, Operation2<F> &op, bool delay = false, VolumeSetMap *reuse = 0, float beta = 1.0);
	void add_op_rollout(std::string in, std::string out, Operation<F> &op, bool delay = false, VolumeSetMap *reuse = 0, float beta = 1.0);

 	void add_volume(std::string name, VolumeShape shape, VolumeSetMap *reuse = 0);
	void add_volume(std::string name, VolumeSet &set);
	bool exists(std::string name);
	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F>> &fast_grads);

 	virtual void forward_dry_run();
 	virtual void forward();
 	virtual void backward();

 	void update(float lr);

 	void scale_grad();
	void clear();
	void clear_grad();

	void init_normal(F mean, F std);
	void init_uniform(F var);

	VolumeSet &input() { return *vin; }
	VolumeSet &output() { return *vout; }


	VolumeShape in_shape;
 	VolumeSetMap volumes;

 	std::vector<TimeOperation*>   operations;
	std::vector<Parametrised<F>*> parameters;
 	VolumeSet *vin, *vout;

	int T;
  int n_param;
};

struct LSTMOperation : public SubVolumeOperation {
	LSTMOperation(VolumeShape in, int kg, int ko, int c, VolumeSetMap *reuse = 0);
	LSTMOperation(VolumeShape in, int kg, int ko, int c, bool rollout, VolumeSetMap *reuse = 0);
	virtual ~LSTMOperation(){}

 	// void forward();
 	// void backward();

	void add_operations(VolumeSetMap *reuse = 0);
	void add_operations_rollout(VolumeSetMap *reuse = 0);

	void share(LSTMOperation &o);

	ConvolutionOperation<F> xi, hi; //input gate
	ConvolutionOperation<F> xr, hr; //remember gate (forget gates dont make sense!)
	ConvolutionOperation<F> xs, hs; //cell input
	ConvolutionOperation<F> xo, ho;//, co; //output gate
	ConvolutionOperation<F> cc;

	GateOperation<F>    gate;   //for gating
	SigmoidOperation<F> sig;
	TanhOperation<F>    tan;

};


struct LSTMShiftOperation : public SubVolumeOperation {
	LSTMShiftOperation(VolumeShape in, int kg, int ko, int c, int dx, int dy, VolumeSetMap *reuse = 0);
	virtual ~LSTMShiftOperation(){}
 	// void forward();
 	// void backward();

	void add_operations(VolumeSetMap *reuse = 0);


	ConvolutionShiftOperation<F> xi, hi; //input gate
	ConvolutionShiftOperation<F> xr, hr; //remember gate (forget gates dont make sense!)
	ConvolutionShiftOperation<F> xs, hs; //cell input
	ConvolutionShiftOperation<F> xo, ho;//, co; //output gate
	ConvolutionOperation<F> cc;

	GateOperation<F>    gate;   //for gating
	SigmoidOperation<F> sig;
	TanhOperation<F>    tan;

};


struct HWOperation : public SubVolumeOperation {
  HWOperation(VolumeShape in, int kw, VolumeSetMap *reuse = 0);
  virtual ~HWOperation(){}

  // void forward();
  // void backward();
  
  void add_hw_operations(VolumeSetMap *reuse = 0);
    
  void share(HWOperation &o);

  ConvolutionOperation<F> hg, hh, xg, xh;
  
  GateOperation<F>    gate;   //for gating
  SigmoidOperation<F> sig;
  TanhOperation<F>    tan;
};


struct VLSTMOperation : public VolumeOperation {
	VLSTMOperation();
	VLSTMOperation(VolumeShape shape, int kg, int ko, int c, VolumeSetMap &vsm);

	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);

	void forward_dry_run(Volume &in, Volume &out);
	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &fast_params, std::vector<CudaPtr<F>> &grads, std::vector<CudaPtr<F>> &fast_grads);
	VolumeShape output_shape(VolumeShape s);

	void prepare();
	void sharing();
	void clear();
	void clear_grad();
	void init_normal(F mean, F std);
	void init_uniform(F var);
	void update(float lr);
	void describe(std::ostream &out) { out << "vlstm " << kg << " " << ko << " " << c; }

	int kg, ko;
	int c;

	std::vector<SubVolumeOperation*> operations;
};

struct HWVOperation : public VLSTMOperation {
  HWVOperation(VolumeShape shape, int kw, VolumeSetMap &vsm);
  void describe(std::ostream &out) { out << "vhw kw:" << kw << " c:" << c; }

  int kw;
};

struct UniVLSTMOperation : public VLSTMOperation {
	UniVLSTMOperation(VolumeShape shape, int kg, int ko, int c, VolumeSetMap &vsm);

	void describe(std::ostream &out) { out << "uni vlstm " << kg << " " << ko << " " << c; }
	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);
};

#endif
