#include "learningsystem.h"
#include "network.h"
#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "log.h"
#include "divide.h"
#include "trainer.h"
#include "util.h"
#include "utilvr.h"

using namespace std;

void test(int argc, char **argv) {
	srand(time(0));
	Handler::set_device(0);

	string exp_dir("exp-uni/");
	Log logger(exp_dir + "log.txt");

    
    int img_size(10*10);
	vector<float> img_data(img_size);

	// int img_w = img.w();
	// int img_h = img.h();
	// int img_c = img.c();

	int img_w = 200;
	int img_h = 200;
	int img_c = 6;
	int train_n = 120;


	cout << "whc: " << img_w << " " << img_h << " " << img_c << endl;
	// cout << img.data()[10] << endl;



	//VolumeShape shape{100, 1, 512, 512};


	//int kg(3), ko(3), c(1);
	// VolumeShape train_shape{train_n, img_c + 44, img_w, img_h};
	VolumeShape train_shape{train_n, img_c, img_w, img_h};
	VolumeShape target_shape{train_n, img_c, img_w, img_h};

	int kg(7), ko(7), c(1);


	VolumeNetwork net(train_shape);

	net.add_fc(32);
	net.add_tanh();
	net.add_univlstm(7, 7, 16);
	net.add_univlstm(7, 7, 32);
    net.add_fc(64);
	net.add_tanh();
    net.add_fc(img_c);
	net.finish();

    net.init_uniform(.1);
	cout << "n params: " << net.param_vec.n << endl;
	// return 1;

	if (argc > 1) {
	  net.load(argv[1]);
	}

	logger << "begin net description\n";
	logger << "input volume shape " << train_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";


	//Fast-weight network
	TensorShape action_input{train_n, 3+41, 1, 1};

	Network<float> fastweight_net(action_input);
	// fastweight_net.add_conv(16, 1, 1);
	// fastweight_net.add_tanh();
	// fastweight_net.add_conv(32, 1, 1);
	// fastweight_net.add_tanh();
	fastweight_net.add_conv(64, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(32, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(16, 1, 1);
	fastweight_net.add_tanh();

	// fastweight_net.add_conv(net.fast_param_vec.n / train_n, 1, 1);

	fastweight_net.add_tanh();
	fastweight_net.finish();

	// fastweight_net.init_uniform(.1);
	fastweight_net.init_normal(.0, .1);

	logger << "begin fastweight description\n";
	logger << "input volume shape " << train_shape << "\n";
	fastweight_net.describe(logger.file);
	logger << "end description\n";

	int epoch(0);
	float last_loss = 9999999.;

	int n_sums(50); // marijn trick vars
	int sum_counter(0);
	int burnin(50);


	Volume input(train_shape), target(target_shape);

	Trainer trainer(net.param_vec.n, .01, .0000001, 400, .1, 50);
	// Trainer fast_trainer(fastweight_net.n_params, .00001, .0000001, 100);
	Trainer fast_trainer(fastweight_net.n_params, .01, .00001, 400, .1, 50);


	while (true) {
		ostringstream epoch_path;
		epoch_path << exp_dir << epoch << "-";

        ////random_next_step_subvolume(db, net.input(), target, fastweight_net.input());

        // random_next_step_subvolume_added_info(db, net.input(), target, fastweight_net.input());
		// if (epoch % 100 == 0)
		// cout << "fastweight input: " << fastweight_net.input().shape() << " " << fastweight_net.input().to_vector() << endl;
		Timer fasttimer;

		// fastweight_net.forward();

		cout << "fast forward took:" << fasttimer.since() << endl;
		// cout << "fastweight output: ";
		// print_last(fastweight_net.output().to_vector(), 20);
		// cout << fastweight_net.output().to_vector() << endl;
		// cout << fastweight_net.input().to_vector() << endl;

		// net.set_fast_weights(fastweight_net.output());

		//cout << net.fast_param_vec.to_vector() << endl;
		// cout << net.param_vec.to_vector() << endl;

	    Timer total_timer;

		Timer ftimer;
		net.forward();
		cout << "forward took:" << ftimer.since() << endl;

		if (epoch % 200 == 0) {
			net.input().draw_slice(epoch_path.str() + "input_last.png",	train_n-1);
			net.input().draw_slice(epoch_path.str() + "input_middle.png",	train_n / 2);
			net.output().draw_slice(epoch_path.str() + "output_middle.png",train_n / 2);
			net.output().draw_slice(epoch_path.str() + "output_last.png",train_n - 1);
			cout << "output/target:" << endl;
			print_wide(net.output().to_vector(), 30);
			print_wide(target.to_vector(), 30);
			target.draw_slice(epoch_path.str() + "target_middle.png",train_n/2);
			target.draw_slice(epoch_path.str() + "target_last.png",train_n-1);
			net.save(exp_dir + "volnet.net");
			fastweight_net.save(exp_dir + "fastnet.net");
		}

		float loss = net.calculate_loss(target);
		logger << "epoch: " << epoch << ": loss " << sqrt(loss / target_shape.size()) << "\n";
		last_loss = loss;

		Timer timer;
		// cout << last(net.volumes)->diff.to_vector() << endl;
		net.backward();
		net.grad_vec *= 1.0 / target_shape.size();
		// net.fast_grad_vec *= 1.0 / train_shape.size();
		cout << "backward took:" << timer.since() << "\n\n";
		cout << "grad: " << endl;
		// print_wide(net.grad_vec.to_vector(), 20);
		trainer.update(&net.param_vec, net.grad_vec);

		// net.get_fast_grads(fastweight_net.output_grad());
		// fastweight_net.backward();
		// fast_trainer.update(&fastweight_net.param_vec, fastweight_net.grad_vec);
		// cout << fastweight_net.output_grad().to_vector() << endl;

		// ((LSTMOperation*)((VLSTMOperation*)net.operations[0])->operations[0])->xi.filter_bank.draw_filterbank("filters.png");

		// ((LSTMShiftOperation*)((UniVLSTMOperation*)net.operations[0])->operations[1])->xi.filter_bank.draw_filterbank("filters2.png");
		// ((LSTMOperation*)((VLSTMOperation*)net.operations[0])->operations[0])->xi.filter_bank.draw_filterbank("filters.png");
		// ((LSTMOperation*)((VLSTMOperation*)net.operations[0])->operations[1])->xi.filter_bank.draw_filterbank("filters2.png");

		++epoch;
		cout << "epoch time: "  << total_timer.since() << endl;
		// return 0;
	}

	cudaDeviceSynchronize();
}



