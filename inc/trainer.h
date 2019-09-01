#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "cudavec.h"
#include "volumenetwork.h"
#include <cmath>

//RMS averaged gradient with momentum
struct Trainer {
  Trainer(int n_params, float start_lr, float end_lr, float half_time, float momentum = .9);

	void update(CudaVec *param, CudaVec &grad);
	float lr();

	CudaVec a,b,c,d,e, rmse;

	float var_decay;
	float mean_decay;
	float eps;
	float base_lr, end_lr, lr_decay;
	float var_decay_bias, mean_decay_bias;

	bool first;
};

#endif
