#include "trainer.h"
#include "log.h"

Trainer::Trainer(int param_n, float start_lr_, float end_lr_, float half_time, float momentum) :
a(param_n),
b(param_n),
c(param_n),
d(param_n),
e(param_n),
rmse(param_n),
var_decay(0.9),
mean_decay(momentum),
eps(.00001),
base_lr(start_lr_ - end_lr_),
end_lr(end_lr_),
lr_decay(pow(.5, 1.0 / half_time)),
var_decay_bias(1.0),
mean_decay_bias(1.0),
first(true)
{

}

float Trainer::lr() {
	return end_lr + base_lr;
}

void Trainer::update(CudaVec *param, CudaVec &grad) {
	float lr_ = lr();
	
	var_decay_bias *= var_decay;
	mean_decay_bias *= mean_decay;
	base_lr *= lr_decay;

	//tmp
	/*	c = grad;
	c *= lr_;
	*param -= c;
	std::cout << c.to_vector() << std::endl;
	*/

       
	//variance
	a = grad;
	a *= a;
	a *= (1.0 - var_decay);
	rmse *= var_decay;
	rmse += a;

	//momentum (snd moment)
	c = grad;
	c *= (1.0 - mean_decay);
	e *= mean_decay;
	e += c; //e = momentum grad

	b = rmse;
	b *= 1.0 / (1.0 - var_decay_bias);//necessary for initial steps
	b.sqrt();
	b += eps; //to prevent divide by zero

	d = e;
	// d *= 1.0 / (1.0 - mean_decay_bias);
	d /= b;
	d.clip(4);// 4 times std
	//update
	//c.clip(1.);
	d *= lr_;
	*param += d;

	// *param *= .99 * lr_; //l2


	first = false;
	

	// print_last(grad.to_vector(), 10);
	// print_last(rmse.to_vector(), 10);
	// print_last(c.to_vector(), 10);
	// print_last(d.to_vector(), 10);
	// print_last(e.to_vector(), 10);
	// print_last((*param).to_vector(), 10);
}

// void Trainer::update(CudaVec *param, CudaVec &grad) {
// 	float lr_ = lr();
// 	base_lr *= lr_decay;

// 	a = grad;
// 	a *= a;
// 	if (first) rmse = a;

// 	rmse *= decay;
// 	a *= (1.0 - decay);
// 	rmse += a;

// 	b = rmse;
// 	b.sqrt();
// 	b += eps;

// 	c = grad;
// 	c /= b;

// 	//SGD
// 	// net.grad *= .00001;
// 	// net.param += net.grad;

// 	//Marijn Trick

// 	//d = c;
// 	//d *= (1.0 - mean_decay);
// 	//e *= mean_decay;
// 	//e += d;

// 	//d = e;
// 	//d.abs();
// 	//c *= d;

// 	//Marijn Trick 2

// 	// if (epoch >= burnin) {
// 	//   d = param;
// 	//   d *= (1.0 / n_sums);
// 	//   e += d;
// 	//   ++sum_counter;

// 	//   if (sum_counter == n_sums) {
// 	//     param = e;
// 	//     e.zero();
// 	//     c.zero();
// 	//     sum_counter = 0;
// 	//     save("mean.net");
// 	//   }

// 	// }

// 	//Momentum

// 	//d = c;

// 	c *= (1.0 - mean_decay);
// 	e *= mean_decay;
// 	e += c;
// 	d = e;
// 	// c = e;

// 	//update
// 	//c.clip(1.);
// 	d *= lr_;
// 	*param += d;

// 	// *param *= .99 * lr_;

// 	first = false;


// 	// print_last(grad.to_vector(), 10);
// 	// print_last(rmse.to_vector(), 10);
// 	// print_last(c.to_vector(), 10);
// 	// print_last(d.to_vector(), 10);
// 	// print_last(e.to_vector(), 10);
// 	// print_last((*param).to_vector(), 10);
// }
