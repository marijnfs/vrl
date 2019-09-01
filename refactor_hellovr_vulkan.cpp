//========= Copyright Valve Corporation ============//

#include <string>
#include <cstdlib>
#include <inttypes.h>
#include <openvr.h>
#include <deque>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "shared/lodepng.h"
#include "shared/Matrices.h"
#include "shared/pathtools.h"

#include "img.h"
#include "util.h"
#include "utilvr.h"
#include "global.h"
#include "vulkansystem.h"
#include "vrsystem.h"
#include "windowsystem.h"
#include "flywheel.h"

#include "learningsystem.h"
#include "volumenetwork.h"
#include "network.h"
#include "trainer.h"
#include "saveh5.h"

using namespace std;

enum Orientation {
  Horizontal = 0,
  Vertical = 1,
  Laying = 2,
  Null = 3
};

struct ExperimentStep {
  Orientation orientation = Horizontal;
  float xdir = 0;
  float ydir = 0;
  float zdir = 0;
  int n_clicks = 0;

  float long_side = 0;
  float short_side = 0;

  ExperimentStep(){}
  ExperimentStep(Orientation o, float xdir_, float ydir_, float zdir_, int n_clicks_, float long_side_, float short_side_) :
    orientation(o), xdir(xdir_), ydir(ydir_), zdir(zdir_), n_clicks(n_clicks_),
    long_side(long_side_), short_side(short_side_)
  {}
};

struct FittsWorld {
  Scene &scene;
  int click = 0;
  int step = 0;
  int choice = 0;
  
  vector<ExperimentStep> steps;
  
  

  FittsWorld(Scene &scene_) : scene(scene_) {
    srand(123123);
    init_experiments();
    init();
  }

 
  void init_experiments() {
    //ExperimentStep(Orientation o, float xdir_, float ydir_, float zdir_, int n_clicks_, float long_side_, float short_side_) :
    vector<float> multipliers = {1.0, 2.0, 4.0};
    int n_clicks(10);
    float long_side(.2);
    float short_side(.01);
    float dist(.02);
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Vertical, dist * m + short_side * m2, 0, 0, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Vertical, 0., 0, dist * m + short_side * m2, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Vertical, dist * m + short_side * m2, 0, dist * m + short_side * m2, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Vertical, dist * m + short_side * m2, 0, -dist * m - short_side * m2, n_clicks, long_side, short_side * m2));
    
    ////Horizontal
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Horizontal, 0, dist * m + short_side * m2, 0, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Horizontal, 0., 0, dist * m + short_side * m2, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Horizontal, 0, dist * m + short_side * m2, dist * m + short_side * m2, n_clicks, long_side, short_side * m2));

    for (auto m : multipliers)
      for (auto m2 : multipliers)
      steps.push_back(ExperimentStep(Horizontal, 0, dist * m + short_side * m2, -dist * m - short_side * m2, n_clicks, long_side, short_side * m2));
    
    //Laying
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Laying, dist * m + short_side * m2, 0, 0, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Laying, 0., dist * m + short_side * m2, 0, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
        steps.push_back(ExperimentStep(Laying, dist * m + short_side * m2, dist * m + short_side * m2, 0, n_clicks, long_side, short_side * m2));
    
    for (auto m : multipliers)
      for (auto m2 : multipliers)
      steps.push_back(ExperimentStep(Laying, dist * m + short_side * m2, -dist * m - short_side * m2, 0, n_clicks, long_side, short_side * m2));
    random_shuffle(steps.begin(), steps.end());
    
    steps.resize(steps.size() / 2);///TODO
  }
    
  void init() {
    cout << "Fitts World INIT" << endl;
    //scene.add_canvas("test");
    scene.add_hmd();
    scene.add_object("controller", new Controller(true));
    //scene.set_pos("test", Pos(1, 1, 1));

    scene.add_variable("step", new FreeVariable());
    scene.add_variable("click", new FreeVariable());

    scene.add_variable("orientation", new FreeVariable());
    scene.add_variable("xdir", new FreeVariable());
    scene.add_variable("ydir", new FreeVariable());
    scene.add_variable("zdir", new FreeVariable());

    scene.add_variable("long_side", new FreeVariable());
    scene.add_variable("short_side", new FreeVariable());
    
    scene.add_variable("choice", new FreeVariable());

    scene.add_variable("start", new MarkVariable());
    scene.add_variable("end", new MarkVariable());
    
    scene.register_function("on_in_box", std::bind(&FittsWorld::on_in_box, *this));
    scene.register_function("on_start", std::bind(&FittsWorld::on_start, *this));
    scene.add_trigger(new ClickTrigger(scene("controller")), "on_start");

    scene.add_box("startbox");
    scene.set_pos("startbox", Pos(0, .9, -.05));
    scene.find<Box>("startbox").set_dim(.05, .05, .05);
    scene.find<Box>("startbox").set_texture("white-checker.png");

    cout << "Fitts World Done INIT" << endl;
  }

  void on_in_box() {
    //cout << "ON IN BOX" << endl;

    if (scene.find<Controller>("controller").clicked == true) {
      scene.clear_scene();
      scene.set_reward(1);

      scene.add_trigger(new NextTrigger(), "on_start");
      scene.variable<MarkVariable>("end").set_value(1);
      //scene.end_recording();
      //scene.clear_objects();
      //scene.clear_triggers();
      //scene.add_trigger(new ClickTrigger(), "on_start");
    }
  }
  
  void on_start() {
    click++;
    if (click >= steps[step].n_clicks) {
      step++;
      cout << step << "/" << steps.size() << endl;
      click = 0;
    }
    if (step >= steps.size()) {
      scene.stop = true;
      return;
    }

    ExperimentStep &cur_step(steps[step]);

    int new_choice = rand() % 3;
    while (new_choice == choice)
      new_choice = rand() % 3;
    choice = new_choice;

    //store experiment vars;
    scene.variable<FreeVariable>("step").set_value(step);
    scene.variable<FreeVariable>("click").set_value(click);
    scene.variable<FreeVariable>("orientation").set_value(cur_step.orientation);

    scene.variable<FreeVariable>("long_side").set_value(cur_step.long_side);
    scene.variable<FreeVariable>("short_side").set_value(cur_step.short_side);

    scene.variable<FreeVariable>("xdir").set_value(cur_step.xdir);
    scene.variable<FreeVariable>("ydir").set_value(cur_step.ydir);
    scene.variable<FreeVariable>("zdir").set_value(cur_step.zdir);

    scene.variable<FreeVariable>("choice").set_value(choice);
    scene.variable<MarkVariable>("start").set_value(1);
    //scene.variable<FreeVariable>("start").set_value(1);
        
    scene.set_reward(0);
    scene.clear_objects();
    scene.clear_triggers();

    vector<string> boxes = {"box1", "box2", "box3"};

    float long_side(cur_step.long_side), short_side(cur_step.short_side);
    
    float x(0);
    float y(.9);
    float z(-.05);
    
    float box_width(cur_step.orientation == Horizontal ? long_side : short_side);
    float box_height(cur_step.orientation == Vertical ? long_side : short_side);
    float box_depth(cur_step.orientation == Laying ? long_side : short_side);
    
    scene.add_box("box1");
    scene.set_pos("box1", Pos(x - cur_step.xdir, y - cur_step.ydir, z - cur_step.zdir));
    scene.find<Box>("box1").set_dim(box_width, box_height, box_depth);
    scene.find<Box>("box1").set_texture("white-checker.png");
    
    scene.add_box("box2");
    scene.set_pos("box2", Pos(x, y, z));
    scene.find<Box>("box2").set_dim(box_width, box_height, box_depth);
    scene.find<Box>("box2").set_texture("white-checker.png");
    
    scene.add_box("box3");
    scene.set_pos("box3", Pos(x + cur_step.xdir, y + cur_step.ydir, z + cur_step.zdir));
    scene.find<Box>("box3").set_dim(box_width, box_height, box_depth);
    scene.find<Box>("box3").set_texture("white-checker.png");

    scene.find<Box>(boxes[choice]).set_texture("blue-checker.png");        
    scene.add_trigger(new InBoxTrigger(scene(boxes[choice]), scene("controller")), "on_in_box");
    
    scene.set_reward(0);
    scene.start_recording();
    
  }
};

 
int record(string filename) {
  if (exists(filename))
    throw StringException("file already exists");

  auto &ws = Global::ws();
  auto &vr = Global::vr();
  auto &vk = Global::vk();

  
  vr.setup();
  ws.setup();
  vk.setup();
  
  //preloading images
  ImageFlywheel::preload();

  auto &scene = Global::scene();

  //FittsWorld world(scene);
  Script world_script;
  world_script.run("fittsworld.lua");
  vk.end_submit_cmd();
  
  
  Timer a_timer(1./90);
  uint i(0);
  Recording recording;

  while (!scene.stop) {
    //cout << i << endl;
    vr.update_track_pose();
    scene.step();
    scene.snap(&recording);

    vr.render(scene);
    vr.wait_frame();

    //cout << "elapsed: " << a_timer.elapsed() << endl;
    //if (a_timer.elapsed() > 60.) {
      // recording.save(filename, scene);
    // a_timer.start();
    // }
    //vr.request_poses();
    //a_timer.wait();
  }

  cout << "writing: " << endl;
  recording.save(filename, scene);
  cout << "done: " << endl;
  recording.release();
  Global::shutdown();
  return 0;
}

int test() {
  auto img_data_vec = load_vec<float>("/home/marijnfs/data/allimg.data");
  auto act_data_vec = load_vec<float>("/home/marijnfs/data/allact.data");
  auto obs_data_vec = load_vec<float>("/home/marijnfs/data/allobs.data");

  /*
  auto img_data_vec2 = img_data_vec;
  for (int n(0); n < 50; ++n)
    for (int c(0); c < 6; ++c)
      for (int x = 0; x < 1680 * 1512; ++x)
        img_data_vec[(n * 6 + c) * 1680 * 1512 + x] = img_data_vec2[n * 1680 * 1512 * 6 + x * 6 + c];
  */
        
  int act_dim = 13;
  int obs_dim = 6;
  int aggr_dim = 32;
  int vis_dim = 16;
  
  Volume img_data(VolumeShape{50, 6, VIVE_WIDTH, VIVE_HEIGHT});
  Volume act_data(VolumeShape{50, act_dim, 1, 1});
  Volume obs_data(VolumeShape{50, obs_dim, 1, 1});

  cout << img_data_vec.size() << " " << img_data.size() << endl;
  cout << img_data_vec[0 * 6 * 1680 * 1512 + 450] << " " <<
    img_data_vec[10 * 6 * 1680 * 1512 + 450] << " " <<
    img_data_vec[20 * 6 * 1680 * 1512 + 450] << " " << endl;

  img_data.from_vector(img_data_vec);
  act_data.from_vector(act_data_vec);
  obs_data.from_vector(obs_data_vec);

  VolumeNetwork vis_net(VolumeShape{50, 6, VIVE_WIDTH, VIVE_HEIGHT});
  auto base_pool = new PoolingOperation<F>(4, 4);
  cout << vis_net.output_shape().c << endl;
  
  auto conv1 = new ConvolutionOperation<F>(vis_net.output_shape().c, 8, 5, 5);
  cout << "bla" << endl;
  auto pool1 = new PoolingOperation<F>(4, 4);
  vis_net.add_slicewise(base_pool);
  vis_net.add_slicewise(conv1);
  //vis_net.add_tanh();
  cout << vis_net.output_shape() << endl;
  vis_net.add_slicewise(pool1);
  cout << vis_net.output_shape().tensor_shape() << endl;
  auto squash = new SquashOperation<F>(vis_net.output_shape().tensor_shape(), vis_dim);

  vis_net.add_slicewise(squash);
  cout << "added slicewise" << endl;
  //vis_net.add_tanh();
  vis_net.finish();
  vis_net.init_normal(.0, .1);

  VolumeNetwork aggr_net(VolumeShape{50, 16, 1, 1});
  aggr_net.add_univlstm(1, 1, act_dim);
  aggr_net.finish();
  aggr_net.init_normal(.0, .1);
  
  cout << vis_net.volumes.size() << endl;

  float start_lr = .01;
  float end_lr = .00001;
  float half_time = 200;
  float clip_val = .1;
  float avg_set_time = 50;
  Trainer vis_trainer(vis_net.param_vec.n, start_lr, end_lr, half_time, clip_val, avg_set_time);
  
  while (true) {
    vis_net.input().from_volume(img_data);
    vis_net.forward();
    
    //vis_net.volumes[2]->x.draw_slice("test1.png", 3, 2);
    //vis_net.volumes[2]->x.draw_slice("test2.png", 40, 2);
    copy_gpu_to_gpu(vis_net.output().data(), aggr_net.input().data(), vis_net.output().size());
    aggr_net.forward();
    
    float loss = aggr_net.calculate_loss(act_data);
    aggr_net.backward();

    copy_gpu_to_gpu(aggr_net.input_grad().data(), vis_net.output_grad().data(), aggr_net.input_grad().size());
    vis_net.backward();
    
    /*
    auto v1 = aggr_net.output().to_vector();
    auto v2 = act_data.to_vector();
    
    for (int i(0); i < v1.size(); ++i) {
      if ( i % act_dim == 0) cout << endl;
      cout << v1[i] << ":" << v2[i] << " ";
    }
    */
    //return -1;
    cout << endl;
    cout << "loss: " << loss << endl;

    aggr_net.grad_vec *= 1.0 / 50000;
    aggr_net.param_vec += aggr_net.grad_vec;

    vis_net.grad_vec *= 1.0 / 100000;
    vis_net.param_vec += vis_net.grad_vec;
    //vis_trainer.update(&vis_net.param_vec, vis_net.grad_vec);
  }
}

int replay(string filename) {
  if (!exists(filename))
    throw StringException("file doesnt exist");

  Global::inst().HEADLESS = true;
  
  auto &ws = Global::ws();
  auto &vr = Global::vr();
  auto &vk = Global::vk();

  
  vr.setup();
  ws.setup();
  vk.setup();

  //preloading images
  ImageFlywheel::preload();

  auto &scene = Global::scene();
  //FittsWorld world(scene);
  Script world_script;
  world_script.run("fittsworld.lua");
  vk.end_submit_cmd();
  

  Pose cur_pose, last_pose;

  Timer a_timer(1./90);
  uint i(0);
  Recording recording;
  recording.load(filename, &scene);
  cout << "recording size: " << recording.size() << endl;
  /*
  for (auto o : scene.objects)
    cout << o.first << " " << scene.names[o.second->nameid] << endl;

  for (auto v : scene.variables)
    cout << v.first << " " << v.second->val << " " << scene.names[v.second->nameid] << endl;
  for (auto t : scene.triggers)
    cout << scene.names[t->function_nameid] << endl;
  */

  int T(50);
  std::vector<float> nimg(3 * 2 * VIVE_WIDTH * VIVE_HEIGHT);
  std::vector<float> all_img(nimg.size() * T);
  std::vector<float> all_action((3 + 4 + 4 + 1 + 1) * T);
  std::vector<float> all_obs(6 * T);
  
  int t(0);
  //for (int i(1000); i < 1000+T; ++i, ++t) {
  for (int i(3000); i < recording.size(); ++i, ++t) {
      recording.load_scene(i, &scene);
      
      vr.hmd_pose = Matrix4(scene.find<HMD>("hmd").to_mat4());
      cout << "scene " << i << " items: " << scene.objects.size() << endl;
      bool headless(true);

      //
      ////vr.render(scene, &img);
      ////vr.render(scene, headless);
      vr.render(scene, false);
      vr.wait_frame();
      //std::vector<float> nimg(img.begin(), img.end());
      //normalize_mt(&img);
      //cout << nimg.size() << endl;

      //write_img1c("bla2.png", VIVE_WIDTH, VIVE_HEIGHT, &nimg[0]);
      //copy(nimg.begin(), nimg.end(), all_img.begin() + nimg.size() * t);
      last_pose = cur_pose;
      cur_pose.from_scene(scene);
      if (t == 0) last_pose = cur_pose;
      
      Action action(last_pose, cur_pose);
      
      auto a_vec = action.to_vector();
      auto o_vec = cur_pose.to_obs_vector();
      
      //copy(a_vec.begin(), a_vec.end(), all_action.begin() + 13*t);
      //copy(o_vec.begin(), o_vec.end(), all_obs.begin() + 6*t);
      cout << a_vec << endl << o_vec << endl;
  }

  ofstream img_out("allimg.data", ios::binary);
  size_t s = all_img.size();
  img_out.write((char*)&s, sizeof(s));
  img_out.write((char*)&all_img[0], sizeof(float) * all_img.size());

  ofstream act_out("allact.data", ios::binary);
  s = all_action.size();
  act_out.write((char*)&s, sizeof(s));
  act_out.write((char*)&all_action[0], sizeof(float) * all_action.size());


  ofstream obs_out("allobs.data", ios::binary);
  s = all_obs.size();
  obs_out.write((char*)&s, sizeof(s));
  obs_out.write((char*)&all_obs[0], sizeof(float) * all_obs.size());
  
  /*
  while (i < recording.size()) {
    //cout << i << endl;
    //vr.update_track_pose();
    //scene.step();
    //scene.snap(&recording);

    recording.load_scene(i, &scene);
    vr.hmd_pose = Matrix4(scene.find<HMD>("hmd").to_mat4());
    cout << "scene " << i << " items: " << scene.objects.size() << endl;
    vr.render(scene);
    vr.wait_frame();
    ++i;
    //vr.request_poses();
    //a_timer.wait();
    }*/

  Global::shutdown();
  return 0;
}

void test_setup_networks(VolumeNetwork &vis_net, VolumeNetwork &act_net, int vis_dim, int act_dim) {
  vis_net.add_slicewise(new PoolingOperation<F>(2, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 4, 3, 3));
  vis_net.add_slicewise(new PoolingOperation<F>(4, 4, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 8, 3, 3));
  vis_net.add_tanh();
  vis_net.add_slicewise(new PoolingOperation<F>(2, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 8, 3, 3));
  vis_net.add_tanh();
  vis_net.add_slicewise(new PoolingOperation<F>(2, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 8, 3, 3));
  vis_net.add_tanh();

  vis_net.add_slicewise(new PoolingOperation<F>(2, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 16, 5, 5));
  vis_net.add_tanh();
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 16, 5, 5));
  vis_net.add_tanh();

  vis_net.add_slicewise(new PoolingOperation<F>(2, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 32, 5, 5));
  vis_net.add_tanh();
  vis_net.add_slicewise(new ConvolutionOperation<F>(vis_net.output_shape().c, 32, 5, 5));
  vis_net.add_tanh();

  //vis_net.add_univlstm(5, 5, 64);
  
  //auto squash = new SquashOperation<F>(vis_net.output_shape().tensor_shape(), vis_dim);
  auto squash = new SquashOperation<F>(vis_net.output_shape().tensor_shape(), vis_dim);
  
  vis_net.add_slicewise(squash);
  vis_net.add_tanh();
  vis_net.add_fc(vis_dim);
  
  //output should be {z, c, 1, 1}
  vis_net.finish();

  act_net.add_fc(16);
  act_net.add_tanh();
  act_net.add_univlstm(1, 1, 32);
  act_net.add_fc(16);
  act_net.add_tanh();
  act_net.add_fc(act_dim);
  
  act_net.finish();
}

void setup_networks(VolumeNetwork &vis_net, VolumeNetwork &aggr_net, Network<F> &actor_net, Network<F> &value_net, VolumeNetwork &q_net, int act_dim, int obs_dim, int vis_dim, int aggr_dim) {
  auto base_pool = new PoolingOperation<F>(2, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
  auto conv1 = new ConvolutionOperation<F>(vis_net.output_shape().c, 8, 5, 5);
  auto pool1 = new PoolingOperation<F>(4, 4, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
  vis_net.add_slicewise(base_pool);
  vis_net.add_slicewise(conv1);
  vis_net.add_tanh();
  vis_net.add_slicewise(pool1);
  vis_net.add_tanh();
  
  auto conv2 = new ConvolutionOperation<F>(vis_net.output_shape().c, 16, 5, 5);
  auto pool2 = new PoolingOperation<F>(4, 4);
  vis_net.add_slicewise(conv2);
  vis_net.add_slicewise(pool2);

  //vis_net.add_univlstm(7, 7, 16);
  //auto squash = new SquashOperation<F>(vis_net.output_shape().tensor_shape(), vis_dim);
  auto squash = new SquashOperation<F>(vis_net.output_shape().tensor_shape(), vis_dim);
 
  vis_net.add_slicewise(squash);
  vis_net.add_tanh();
  //output should be {z, c, 1, 1}
  vis_net.finish();

  aggr_net.add_univlstm(1, 1, 32);
  aggr_net.add_univlstm(1, 1, aggr_dim);
  aggr_net.finish();
  
  actor_net.add_conv(64, 1, 1);
  actor_net.add_relu();
  actor_net.add_conv(64, 1, 1);
  actor_net.add_relu();
  actor_net.add_conv(act_dim, 1, 1);
  actor_net.finish();
  
  value_net.add_conv(64, 1, 1);
  value_net.add_relu();
  value_net.add_conv(64, 1, 1);
  value_net.add_relu();
  value_net.add_conv(1, 1, 1);
  value_net.finish();
  
  q_net.add_univlstm(1, 1, 32);
  q_net.add_univlstm(1, 1, 32);
  q_net.add_fc(1);
  q_net.finish();

}

int learn(string filename) {
  Global::inst().HEADLESS = true;
      
  if (!exists(filename))
    throw StringException("file doesnt exist");

  Handler::cudnn();
  Handler::set_device(0);
  
  //auto &ws = Global::ws();
  auto &vr = Global::vr();
  auto &vk = Global::vk();

  vector<string> scenes{"recordings/user01.rec.fix",
      "recordings/user02.rec.fix",
      "recordings/user03.rec.fix",
      "recordings/user04.rec.fix",
      "recordings/user05.rec.fix",
      "recordings/user06.rec.fix",
      "recordings/user07.rec.fix",
      "recordings/user08.rec.fix",
      "recordings/user09.rec.fix",
      "recordings/user10.rec.fix",
      "recordings/user11.rec.fix",
      "recordings/user12.rec.fix",
      "recordings/user13.rec.fix",
      "recordings/user14.rec.fix",
      "recordings/user15.rec.fix",
      "recordings/user16.rec.fix",
      "recordings/user17.rec.fix",
      "recordings/user18.rec.fix",
      "recordings/user19.rec.fix",
      "recordings/user20.rec.fix",
      "recordings/user21.rec.fix",
      "recordings/user22.rec.fix",
      "recordings/user23.rec.fix",
      "recordings/user24.rec.fix",
      "recordings/user25.rec.fix"};
  
  vr.setup();
  //ws.setup();
  vk.setup();

  //preloading images
  ImageFlywheel::preload();

  auto &scene = Global::scene();
  //FittsWorld world(scene);
  Script world_script;
  world_script.run("fittsworld.lua");
  vk.end_submit_cmd();

  Timer a_timer(1./90);
  
  
  Recording recording;
  recording.load(filename, &scene);
  cout << "recording size: " << recording.size() << endl;

  
  int N = 16;
  int c = 3 * 2; //stereo rgb
  int h = 32;

  int act_dim = 13;
  int obs_dim = 6;
  int vis_dim = 8;
  int aggr_dim = 32;
  
  int width = VIVE_WIDTH;
  int height = VIVE_HEIGHT;
  VolumeShape img_input{N, c, width, height};
  //VolumeShape network_output{N, h, 1, 1};

  //Visual processing network, most cpu intensive
  bool first_grad = false;
  VolumeNetwork vis_net(img_input, first_grad);
 
  //Aggregation Network combining output of visual processing network and state vector
  VolumeShape aggr_input{N, vis_dim + obs_dim, 1, 1};
  //VolumeShape aggr_input{N, vis_dim, 1, 1};
  VolumeNetwork aggr_net(aggr_input);
 
  //TensorShape actor_in{N, aggr_dim, 1, 1};
  TensorShape actor_in{N, vis_dim, 1, 1};
  Network<F> actor_net(actor_in);
  
  TensorShape value_in{N, aggr_dim, 1, 1};
  Network<F> value_net(actor_in);
  
  VolumeShape q_in{N, aggr_dim + act_dim, 1, 1}; //qnet, could be also advantage
  VolumeNetwork q_net(q_in);

  test_setup_networks(vis_net, aggr_net, vis_dim, act_dim);
  //setup_networks(vis_net, aggr_net, actor_net, value_net, q_net, act_dim, obs_dim, vis_dim, aggr_dim);
  
  //initialisation
  float stddev(.15);

  
  vis_net.init_uniform(.15);
  aggr_net.init_uniform(stddev);
  
  ///q_net.init_uniform(std);
  // actor_net.init_uniform(std);
  //value_net.init_uniform(std);

  float vis_start_lr = .01;
  float aggr_start_lr = .01;
  float end_lr = .00001;
  float half_time = 500;
  float clip_val = .1;
  float avg_set_time = 50;
  Trainer vis_trainer(vis_net.param_vec.n, vis_start_lr, end_lr, half_time, clip_val, avg_set_time);
  Trainer aggr_trainer(aggr_net.param_vec.n, aggr_start_lr, end_lr, half_time, clip_val, avg_set_time);

  
  //Trainer actor_trainer(actor_net.param_vec.n, .001, .00001, 200);
  
  Volume action_targets_tensor(VolumeShape{N, act_dim, 1, 1});
  //Tensor<F> action_targets_tensor(N, act_dim, 1, 1);
  vector<float> action_targets(N * act_dim);
  
  
  
  //load if applicable
  /*
  if (exists("vis.net"))
    vis_net.load("vis.net");
  if (exists("aggr.net"))
    aggr_net.load("aggr.net");
  if (exists("q.net"))
    q_net.load("q.net");
  if (exists("actor.net"))
    actor_net.load("actor.net");
  if (exists("value.net"))
    value_net.load("value.net");
  */
  
  int epoch(0);
  std::vector<float> nimg(3 * 2 * VIVE_WIDTH * VIVE_HEIGHT);
  while (true) {
    if (epoch % 10 == 0 && epoch > 0) {
      scene.clear();
      recording.release();
      recording.load(scenes[rand() % scenes.size()], &scene);
    }

    int b = rand() % (recording.size() - N + 2);
    int e = b + N;
    Pose cur_pose, next_pose;

    //l2 adjustment
    //float l2 = .000001;
    //vis_net.param_vec *= 1.0 - l2;
    //aggr_net.param_vec *= 1.0 - l2; 
   
    
    //First setup all inputs for an episode
    int t(0);    
    
    for (int i(b); i < e; ++i, ++t) {
      recording.load_scene(i+1, &scene);
      next_pose.from_scene(scene);
      
      recording.load_scene(i, &scene);
      cur_pose.from_scene(scene);
      
      vr.hmd_pose = Matrix4(scene.find<HMD>("hmd").to_mat4());
      cout << "scene " << i << " items: " << scene.objects.size() << endl;


      //
      ////vr.render(scene, &img);
      ////vr.render(scene, headless);
      bool headless(true);
      vr.render(scene, headless, &nimg);
      
      /*stringstream oss;
      oss << "images/left" << i << ".png";
      stringstream oss2;
      oss2 << "images/right" << i << ".png";
      write_img1c(oss.str(), VIVE_WIDTH, VIVE_HEIGHT, &nimg[0] + VIVE_WIDTH * VIVE_HEIGHT);
      write_img1c(oss2.str(), VIVE_WIDTH, VIVE_HEIGHT, &nimg[0] + VIVE_WIDTH * VIVE_HEIGHT * 4);
      */
      
      //vr.wait_frame();
      //std::vector<float> nimg(img.begin(), img.end());
      //normalize_mt(&img);
      //cout << nimg.size() << endl;

      //write_img1c("bla2.png", VIVE_WIDTH, VIVE_HEIGHT, &nimg[0]);
      //write_img1c("bla3.png", VIVE_WIDTH, VIVE_HEIGHT, &nimg[VIVE_WIDTH * VIVE_HEIGHT * 3]);
      copy_cpu_to_gpu<float>(&nimg[0], vis_net.input().data(t), nimg.size());

      //cout << nimg << endl;
      
      //vis_net.input().draw_slice("bla.png", t, 0);
      //normalise_fast(vis_net.input().slice(t), nimg.size());
     
      Action action(cur_pose, next_pose);
      
      auto a_vec = action.to_vector();
      auto o_vec = cur_pose.to_obs_vector();

      cout << action << endl;
      cout << "action: " << endl << a_vec << "ovec:" << endl << o_vec << endl;
      cout << "A:" << endl;
      cout << action << endl;
      //if (t) //ignore first action
      copy(a_vec.begin(), a_vec.end(), action_targets.begin() + act_dim * t);
      copy_cpu_to_gpu(&o_vec[0], aggr_net.input().data(t, 0), o_vec.size());
    }

    action_targets_tensor.from_vector(action_targets);
    //run visual network

    auto eval_func = [&]()->float {
      vis_net.forward();
      
      //debug
      
      
      cout << "vis output: " << vis_net.output().to_vector() << endl;
      //aggregator
      for (int t(0); t < N; ++t) {
        //aggregator already has observation data, we add vis output after that
        copy_gpu_to_gpu(vis_net.output().data(t), aggr_net.input().data(t, obs_dim), vis_dim);
      }
      //copy_gpu_to_gpu(vis_net.output().data(), aggr_net.input().data(), vis_net.output().size());

      aggr_net.forward();
      cout << "aggr output: " << aggr_net.output().to_vector() << endl;
      
      //copy aggr to nets
      //*copy_gpu_to_gpu(aggr_net.output().data(), actor_net.input().data, aggr_net.output().size());
      //*copy_gpu_to_gpu(aggr_net.output().data(), value_net.input().data, aggr_net.output().size());
      
      //*actor_net.forward();
      //*value_net.forward(); //not for imitation
      
      auto pred_actions = aggr_net.output().to_vector();
      auto p1 = pred_actions.begin(), p2 = action_targets.begin();
      for (int i(0); i < N; ++i) {
        cout << "actor: ";
        for (int n(0); n < act_dim; ++n, ++p1, ++p2)
          cout << *p1 << " " << *p2 << endl;
      }
      
      //copy actions and aggr to q  //not for imitation //HERE BE SEGFAULT?
      /*for (int t(0); t < N - 1; ++t) {
        copy_gpu_to_gpu(actor_net.output().ptr(t), q_net.input().data(t + 1, 0), act_dim);
        copy_gpu_to_gpu(aggr_net.output().data(t), q_net.input().data(t, act_dim), aggr_dim);
        }
        q_net.forward();     //not for imitation
      */
      
      //====== Imitation backward:
      
      //* actor_net.calculate_loss(action_targets_tensor);
      //cout << "actor action: " << actor_net.output().to_vector() << endl;
      auto loss = aggr_net.calculate_loss(action_targets_tensor);
      
      cout << "actor loss: " << loss << endl;
      //cout << "actor loss: " << actor_net.loss() << endl;
      //* actor_net.backward();
      aggr_net.backward();
    
      //cout << aggr_net.param_vec.to_vector() << endl;
      //cout << aggr_net.grad_vec.to_vector() << endl;
      //return 1;
      //copy_gpu_to_gpu(actor_net.input_grad().data(), aggr_net.output_grad().data(), actor_net.input_grad().size());
      //copy_gpu_to_gpu(actor_net.input_grad().data(), aggr_net.output_grad().data(), actor_net.input_grad().size());
      //aggr_net.backward();

      for (int t(0); t < N; ++t) {
        copy_gpu_to_gpu(aggr_net.input_grad().data(t, obs_dim), vis_net.output_grad().data(t), vis_dim);
      }
      
      //cout << "visnet grad: " << vis_net.output_grad().to_vector() << endl;
      

      
      vis_net.backward();
      //cout << vis_net.volumes[12]->diff.to_vector() << endl;
      //throw "";
      return loss;
    };
    
    //vis_trainer.update(&vis_net.param_vec, vis_net.grad_vec);
    //aggr_trainer.update(&aggr_net.param_vec, aggr_net.grad_vec);
    //CudaVec vis_param_back(vis_net.param_vec);
    //CudaVec aggr_param_back(aggr_net.param_vec);

    eval_func();
    vis_trainer.update(&vis_net.param_vec, vis_net.grad_vec);
    aggr_trainer.update(&aggr_net.param_vec, aggr_net.grad_vec);

    if (false) {
      for (int tt(0); tt < N; ++tt) {
        int layer(0);
        for (auto v : vis_net.volumes) {
          ostringstream oss;
          oss << "images/" << t << "_layer_" << layer << ".png";
          cout << "str: " << oss.str() << endl;
          v->x.draw_slice(oss.str(), tt, 1);
          cout << v->x.shape << endl;
          cout << v->diff.shape << endl;

          if (!layer) {
            ++layer;
            continue;
          }
          ostringstream oss2;
          oss2 << "images/" << t << "_grad_" << layer << ".png";
          v->diff.draw_slice(oss2.str(), tt, 1);
          ++layer;
        }
      }
      throw "";
    }
    /*int layer(0);
    for (auto v : vis_net.volumes) {
      ostringstream oss;
      oss << "images/layer_" << layer << ".png";
      v->x.draw_slice(oss.str(), 10, 1);
      ++layer;
      }*/
    //throw "";
    
    
    
    //for (int n(0); n < 4; ++n) {
    //  eval_func();
    //  if (n != 0) {
    //    vis_net.param_vec = vis_param_back;
    //    aggr_net.param_vec = aggr_param_back;
    //  }
    //  vis_trainer.update(&vis_net.param_vec, vis_net.grad_vec);
    //  aggr_trainer.update(&aggr_net.param_vec, aggr_net.grad_vec);
    // }
    //eval_func();
    
    //auto &v = dynamic_cast<ConvolutionOperation<F>*>(dynamic_cast<SlicewiseOperation*>(vis_net.operations[3])->op)->filter_bank_grad;
    //auto &v2 = vis_net.volumes[3]->diff;


    //auto lstmv = vis_net.volumes[11]->x.to_vector();
    //print_split(lstmv, vis_net.shapes[11].w * vis_net.shapes[11].h);
    //int i(0);

    //cout << "vis:" << endl;
    //auto av = vis_net.output().to_vector();
    //throw "";
    //cout << "vis net:" << v.to_vector() << endl;//to_vector() << endl;

    //aggr_net.grad_vec *= 1.0 / 50000;
    //aggr_net.param_vec += aggr_net.grad_vec;

    //vis_net.grad_vec *= 1.0 / 50000;
    //vis_net.param_vec += vis_net.grad_vec;

    //actor_net.grad_vec *= 1.0 / 50000;
    //actor_net.param_vec += actor_net.grad_vec;

   
    
    //actor_trainer.update(&actor_net.param_vec, actor_net.grad_vec);
    //set action targets
    //run backward    
    //====== Qlearning backward:
    //run forward
    //calculate q targets
    //run backward q -> calculate action update
    //run backward rest
    if (epoch % 100 == 0 && epoch > 0) {
      vis_net.save("vis.net");
      aggr_net.save("aggr.net");
      //q_net.save("q.net");
      //actor_net.save("actor.net");
      //value_net.save("value.net");
    }

    ++epoch;
  }

  return 1;
  Global::shutdown();
  return 0;
}

int convert(string filename) {
  Global::inst().HEADLESS = true;
      
  Handler::cudnn();
  Handler::set_device(0);
  
  //auto &ws = Global::ws();
  auto &vr = Global::vr();
  auto &vk = Global::vk();

  vr.setup();
  //ws.setup();
  vk.setup();

  //preloading images
  ImageFlywheel::preload();

  auto &scene = Global::scene();
  //FittsWorld world(scene);
  Script world_script;
  world_script.run("fittsworld.lua");
  vk.end_submit_cmd();
  
  Timer a_timer(1./90);
  
  
  Recording recording;
  recording.load(filename, &scene);
  cout << "recording size: " << recording.size() << endl;
  
  int N = recording.size();
  int c = 3 * 2; //stereo rgb
  
  int act_dim = 13;
  int obs_dim = 6;
  
  int width = VIVE_WIDTH;
  int height = VIVE_HEIGHT;
  VolumeShape img_input{1, c, width, height};
  //VolumeShape network_output{N, h, 1, 1};

  int F(100);

  cout << "allocating" << endl;

  ostringstream oss2;
  // oss2 << "/home/marijnfs/data/big/vr/" << filename << ".json";
  oss2 << filename << ".json";
  ofstream json_out(oss2.str());
  
  std::vector<float> full_image_data(img_input.size() * F);
  cout << "done" << endl;
  
  int epoch(0);


  int frame_counter(0);
  int sequence_counter(0);

  json_out << "{" << endl;

  bool first_scene(true);
    // std::vector<float> nimg(3 * 2 * VIVE_WIDTH * VIVE_HEIGHT);
  for (int s(0); s < recording.size(); ++s, ++frame_counter) {
    recording.load_scene(s, &scene);
    vr.hmd_pose = Matrix4(scene.find<HMD>("hmd").to_mat4());
    cout << "scene " << s << " items: " << scene.objects.size() << endl;

    bool headless(true);
    // vr.render(scene, headless, &nimg);
    // std::copy(nimg.begin(), nimg.end(), &full_image_data[frame_counter * img_input.size()]);

    if (!first_scene)
      json_out << "," << endl;
    first_scene = false;

    json_out << "\"" << s << "\": {" << endl;
    scene.output_json(json_out);
    json_out << "}" << endl;
    // if ((s+1) % F == 0) {
    //   ostringstream oss;
    //   oss << "/home/marijnfs/data/big/vr/" << filename << ".img_" << sequence_counter << ".h5";
    //   save_h5(full_image_data, oss.str(), vector<int>{F, c, width, height});
    //   fill(full_image_data.begin(), full_image_data.end(), 0);
    //   frame_counter = 0;
    // } else if (s + 1 == recording.size()) {
    //   ostringstream oss;
    //   oss << "/home/marijnfs/data/big/vr/" << filename << ".img_" << sequence_counter << ".h5";
    //   save_h5(full_image_data, oss.str(), vector<int>{frame_counter, c, width, height});
    //   fill(full_image_data.begin(), full_image_data.end(), 0);
    //   frame_counter = 0;
    // }
    
    ++sequence_counter;
    //copy_cpu_to_gpu<float>(&nimg[0], auto_targets.slice(s), nimg.size());
  }
  json_out << "}" << endl;
  return 0;
}

int train_autoencoder() {
  Global::inst().HEADLESS = true;
      
  Handler::cudnn();
  Handler::set_device(0);
  
  //auto &ws = Global::ws();
  auto &vr = Global::vr();
  auto &vk = Global::vk();

  vector<string> scenes{"recordings/user01.rec.fix",
      "recordings/user02.rec.fix",
      "recordings/user03.rec.fix",
      "recordings/user04.rec.fix",
      "recordings/user05.rec.fix",
      "recordings/user06.rec.fix",
      "recordings/user07.rec.fix",
      "recordings/user08.rec.fix",
      "recordings/user09.rec.fix",
      "recordings/user10.rec.fix",
      "recordings/user11.rec.fix",
      "recordings/user12.rec.fix",
      "recordings/user13.rec.fix",
      "recordings/user14.rec.fix",
      "recordings/user15.rec.fix",
      "recordings/user16.rec.fix",
      "recordings/user17.rec.fix",
      "recordings/user18.rec.fix",
      "recordings/user19.rec.fix",
      "recordings/user20.rec.fix",
      "recordings/user21.rec.fix",
      "recordings/user22.rec.fix",
      "recordings/user23.rec.fix",
      "recordings/user24.rec.fix",
      "recordings/user25.rec.fix"};
  
  vr.setup();
  //ws.setup();
  vk.setup();

  //preloading images
  ImageFlywheel::preload();

  auto &scene = Global::scene();
  //FittsWorld world(scene);
  Script world_script;
  world_script.run("fittsworld.lua");
  vk.end_submit_cmd();
  
  Timer a_timer(1./90);
  
  
  Recording recording;
  recording.load(scenes[scenes.size()/2], &scene);
  cout << "recording size: " << recording.size() << endl;

  
  int N = 1;
  int c = 3 * 2; //stereo rgb
  int h = 32;

  int act_dim = 13;
  int obs_dim = 6;
  int vis_dim = 8;
  int aggr_dim = 32;
  
  int width = VIVE_WIDTH;
  int height = VIVE_HEIGHT;
  VolumeShape img_input{N, c, width, height};
  //VolumeShape network_output{N, h, 1, 1};
  
  
  //Visual processing network, most cpu intensive
  bool first_grad = false;

  /*VolumeNetwork auto_net(img_input, first_grad);
  
  auto_net.add_slicewise(new PoolingOperation<F>(8, 8, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
  auto img_shape = auto_net.output_shape();
  auto_net.add_slicewise(new SquashOperation<F>(auto_net.output_shape().tensor_shape(), 256));
  auto_net.add_tanh();
  auto_net.add_slicewise(new ConvolutionOperation<F>(auto_net.output_shape().c, 64, 1, 1));
  auto_net.add_tanh();
  auto_net.add_slicewise(new ConvolutionOperation<F>(auto_net.output_shape().c, 256, 1, 1));
  auto_net.add_tanh();
  cout << "shape size: " << img_shape.size() << endl;
  auto_net.add_slicewise(new ConvolutionOperation<F>(auto_net.output_shape().c, img_shape.size(), 1, 1));
  cout << "finish:" << endl;
  auto_net.finish();
  auto_net.init_uniform(.15);
  */
  Network<float> auto_net(img_input.tensor_shape());
  
  cout << "INPUT" << img_input << endl;
  cout << "INPUT tensor:" << img_input.tensor_shape() << endl;
  //auto_net.add_pool(2, 2);
  
  auto img_shape = auto_net.output_shape();
  //auto_net.add_conv(6, 3, 3);
  //auto_net.add_tanh();
  auto_net.add_split(); //split1
  auto_net.add_conv(24, 3, 3);
  //auto_net.add_relu();

  auto_net.add_split(); //split2
  auto_net.add_conv(24, 3, 3);
  //auto_net.add_relu();
  
  auto_net.add_split(); //split3
  auto_net.add_conv(48, 3, 3);
  //auto_net.add_relu();

  auto_net.add_merge(); //merge3
  auto_net.add_conv(24, 3, 3);
  //auto_net.add_relu();

  auto_net.add_merge(); //merge2
  auto_net.add_conv(24, 3, 3);
  //auto_net.add_relu();
  
  auto_net.add_merge(); //merge1
  //auto_net.add_conv(6, 3, 3);
  //auto_net.add_tanh();
  
  auto_net.finish();
  auto_net.init_uniform(.05);

  Volume img_volume(VolumeShape{img_shape.n, img_shape.c, img_shape.w, img_shape.h});

  
  float auto_start_lr = .01;
  float end_lr = .00001;
  float half_time = 500;
  float clip_val = 1;
  float avg_set_time = 300;
  cout << "param: " << auto_net.param_vec.n << endl;
  Trainer auto_trainer(auto_net.param_vec.n, auto_start_lr, end_lr, half_time, clip_val, avg_set_time);

  
  //if (exists("auto.net"))
  //  auto_net.load("auto.net");

  
  int epoch(0);
  while (true) {
    if (epoch % 10 == 0) {
      scene.clear();
      recording.release();
      recording.load(scenes[rand() % scenes.size()], &scene);
    }

    for (int s(0); s < N; ++s) {
      int b = rand() % recording.size();
      
      std::vector<float> nimg(3 * 2 * VIVE_WIDTH * VIVE_HEIGHT);
      
      recording.load_scene(b, &scene);
      
      vr.hmd_pose = Matrix4(scene.find<HMD>("hmd").to_mat4());
      cout << "scene " << b << " items: " << scene.objects.size() << endl;

      bool headless(true);
      vr.render(scene, headless, &nimg);
      
      copy_cpu_to_gpu<float>(&nimg[0], auto_net.input().ptr(s), nimg.size());
      //copy_cpu_to_gpu<float>(&nimg[0], auto_targets.slice(s), nimg.size());
    }
    
    //run visual network

    auto eval_func = [&]()->float {
      auto_net.forward();
      
      if (epoch % 100 == 0)
        {      
          
          //cout << "str: " << oss.str() << endl;
          //cout << auto_net.output().ptr() << " " << img_volume.data() << " " << img_volume.size() << " " << auto_net.output().size() << endl;
        for (int i(0); i < auto_net.tensors.size(); i++) {
          ostringstream oss;
          oss << "images/" << "output_" << epoch << "_l" << i << ".png";
          auto_net.tensors[i]->x.write_img(oss.str(), 2);
        }
        
        //copy_gpu_to_gpu(auto_net.output().ptr(), img_volume.data(), img_volume.size());
        //img_volume.from_volume(auto_net.output());
        //auto_net.output().draw_slice(oss.str(), 4, 0);
        //img_volume.draw_slice(oss.str(), 4, 0);
      }
     
      if (epoch % 10 == 0)
      {
        ostringstream oss;
        oss << "images/" << "input_" << epoch << ".png";
        cout << "str: " << oss.str() << endl;
        //auto_net.volumes[1]->x.draw_slice(oss.str(), 4, 0);
      }

      //debug
      //auto loss = auto_net.calculate_loss(auto_targets);
      //cout << auto_net.output().to_vector() << endl;
      cout << auto_net.tensors[0]->x.size() << " " << auto_net.output().size() << endl;
      //for (auto a : auto_net.tensors[0]->x.to_vector())
      //  if (a>0)
      //    cout << a << endl;
      cout << "grad" << endl;
      //for (auto a : auto_net.tensors[1]->grad.to_vector())
      //  if (a>0)
      //    cout << a << endl;
      //cout << auto_net.tensors[0]->x.to_vector()[10] << endl << auto_net.output().to_vector()[10] << endl;
      //throw "";


      auto loss = auto_net.calculate_loss(auto_net.tensors[0]->x);
      
      
      cout << "loss: " << loss << endl;
      
      auto_net.backward();
      return loss;
    };
    /*
    auto_trainer.c = auto_net.param_vec;
    
    eval_func();
    auto_net.grad_vec *= .00001;
    auto_net.param_vec += auto_net.grad_vec;
    eval_func();
    
    auto_net.grad_vec *= .00001;
    auto_net.param_vec = auto_trainer.c;
    auto_net.param_vec += auto_net.grad_vec;
    eval_func();

    auto_net.grad_vec *= .00001;
    auto_net.param_vec = auto_trainer.c;
    auto_net.param_vec += auto_net.grad_vec;
    eval_func();
    */

    eval_func();
    //cout << auto_net.grad_vec.to_vector() << endl;


    /*cout << "input:" << endl;
    print_separate(auto_net.tensors[1]->x.to_vector(), 80);
    cout << "grad:" << endl;
    print_separate(auto_net.tensors[1]->grad.to_vector(), 80);
        
    cout << "output:" << endl;
    print_separate(auto_net.output().to_vector(), 80);
    cout << auto_net.param_vec.n << " " << auto_net.grad_vec.n << endl;
    */
    auto_trainer.update(&auto_net.param_vec, auto_net.grad_vec);
    cout << "param: " << auto_net.param_vec.n << endl;
    //print_separate(auto_net.param_vec.to_vector(), 80);
    //cout << auto_net.param_vec.to_vector() << endl;
    /*{
      for (int tt(0); tt < N; ++tt) {
        int layer(0);
        for (auto v : vis_net.volumes) {
          ostringstream oss;
          oss << "images/" << t << "_layer_" << layer << ".png";
          cout << "str: " << oss.str() << endl;
          v->x.draw_slice(oss.str(), tt, 1);
          cout << v->x.shape << endl;
          cout << v->diff.shape << endl;

          if (!layer) {
            ++layer;
            continue;
          }
          ostringstream oss2;
          oss2 << "images/" << t << "_grad_" << layer << ".png";
          v->diff.draw_slice(oss2.str(), tt, 1);
          ++layer;
        }
      }
      throw "";
      }*/
    if (epoch % 100 == 0 && epoch > 0) {
      auto_net.save("auto.net");
    }
    
    ++epoch;
  }

  return 1;
  Global::shutdown();
  return 0;
}


int learn_old(string filename) {
  Global::inst().INVERT_CORRECTION = true;
  cerr << "Warning, Invert correction on!" << endl;
  return learn(filename);
}  

int rollout(string filename) {
  
  Global::inst().HEADLESS = true;
  
  if (!exists(filename))
    throw StringException("file doesnt exist");

  Handler::cudnn();
  Handler::set_device(0);
  
  //auto &ws = Global::ws();
  auto &vr = Global::vr();
  auto &vk = Global::vk();

  vr.setup();
  //ws.setup();
  vk.setup();

  //preloading images
  ImageFlywheel::preload();

  auto &scene = Global::scene();
  Script world_script;
  world_script.run("fittsworld.lua");
  //FittsWorld world(scene);
  vk.end_submit_cmd();
  
  Timer a_timer(1./90);
  Recording recording;
  recording.load(filename, &scene);
  cout << "recording size: " << recording.size() << endl;

  int N = 64;
  int c = 3 * 2; //stereo rgb
  int h = 32;

  int act_dim = 13;
  int obs_dim = 6;
  int vis_dim = 8;
  int aggr_dim = 32;

 
  int width = VIVE_WIDTH;
  int height = VIVE_HEIGHT;

  VolumeShape img_input{N, c, width, height};
  //VolumeShape network_output{N, h, 1, 1};


  //Visual processing network, most cpu intensive
  bool first_grad = false;
  VolumeNetwork vis_net(img_input, first_grad);
 
  //Aggregation Network combining output of visual processing network and state vector
  VolumeShape aggr_input{N, vis_dim + obs_dim, 1, 1};
  //VolumeShape aggr_input{N, vis_dim, 1, 1};
  VolumeNetwork aggr_net(aggr_input);
 
  //TensorShape actor_in{N, aggr_dim, 1, 1};
  TensorShape actor_in{N, vis_dim, 1, 1};
  Network<F> actor_net(actor_in);
  
  TensorShape value_in{N, aggr_dim, 1, 1};
  Network<F> value_net(actor_in);
  
  VolumeShape q_in{N, aggr_dim + act_dim, 1, 1}; //qnet, could be also advantage
  VolumeNetwork q_net(q_in);

  test_setup_networks(vis_net, aggr_net, vis_dim, act_dim);
  //setup_networks(vis_net, aggr_net, actor_net, value_net, q_net, act_dim, obs_dim, vis_dim, aggr_dim);
  
  //initialisation

  //load if applicable
  if (exists("vis.net"))
    vis_net.load("vis.net");
  if (exists("aggr.net"))
    aggr_net.load("aggr.net");
  if (exists("q.net"))
    q_net.load("q.net");
  if (exists("actor.net"))
    actor_net.load("actor.net");
  if (exists("value.net"))
    value_net.load("value.net");

  /*
  recording.load_scene(1000, &scene);
  Pose p1(scene);
  recording.load_scene(1100, &scene);
  Pose p2(scene);
  Action a1(p1, p2);
  auto v = a1.to_vector();
  Action a2(v);
  p1.apply(a2);

  cout << p1 << endl;
 cout << p2 << endl;
  return 1;*/

  std::vector<float> nimg(3 * 2 * VIVE_WIDTH * VIVE_HEIGHT);
  
  int epoch(0);
  while (true) {
    int b = rand() % (recording.size() - N + 1);
    int e = b + N;
    Pose cur_pose, last_pose;

    //First setup all inputs for an episode
    int t(0);
    
    
    recording.load_scene(b, &scene);

    for (int i(b); i < e - 1; ++i, ++t) {      
      //recording.load_scene(i+1, &scene);
      //Pose next_pose;
      //next_pose.from_scene(scene);

      //recording.load_scene(i, &scene);
      
      vr.hmd_pose = Matrix4(scene.find<HMD>("hmd").to_mat4());
      cout << "scene " << i << " items: " << scene.objects.size() << endl;

      bool headless(true);
      //bool headless(false);

      ////vr.render(scene, &img);
      ////vr.render(scene, headless);
      
      vr.render(scene, headless, &nimg);

      //std::vector<float> nimg(img.begin(), img.end());
     //normalize_mt(&img);
      //cout << nimg.size() << endl;

      {
        //oss << "bla" << i << ".png" << endl;
        //write_img1c(oss.str(), VIVE_WIDTH, VIVE_HEIGHT, &nimg[0]);
        copy_cpu_to_gpu<float>(&nimg[0], vis_net.input().slice(t), nimg.size());

        ostringstream oss;
        oss << "images/bla-" << i << ".png";
        vis_net.input().draw_slice(oss.str(), t, 0);
        //normalise_fast(vis_net.input().slice(t), nimg.size());
      }

      
      //added to TEST

      cout << "same pos?: " << cur_pose << endl;
      cur_pose.from_scene(scene);
      
      cout << cur_pose << endl;
      cout << "contr p:" << scene.find<Controller>("controller").p << endl;

      cur_pose.apply_to_scene(scene);
      cout << "contr p2:" << scene.find<Controller>("controller").p << endl;
      cur_pose.from_scene(scene);
      cout << "contr p:" << scene.find<Controller>("controller").p << endl;
      cout << cur_pose << endl;
      cur_pose.apply_to_scene(scene);
      cout << "contr p2:" << scene.find<Controller>("controller").p << endl;
        
      cout << t << " cur: " << cur_pose << endl;
      //Action action(cur_pose, next_pose);
      
      auto o_vec = cur_pose.to_obs_vector();
      
      copy_cpu_to_gpu(&o_vec[0], aggr_net.input().data(t), o_vec.size());
      cout << "ovec: " << o_vec << endl;
      
      //run visual network
      vis_net.forward();

      
      //ostringstream oss;
      //oss << "test-" << t << ".png";
      //vis_net.volumes[7]->x.draw_slice(oss.str(), t, 2);
      copy_gpu_to_gpu(vis_net.output().data(t), aggr_net.input().data(t, obs_dim), vis_dim);
      aggr_net.forward();
      //cout << "agg output: " << aggr_net.output().to_vector() << endl;

      /*
      //copy aggr to value and actor nets
      copy_gpu_to_gpu(aggr_net.output().data(), actor_net.input().data, aggr_net.output().size());
      copy_gpu_to_gpu(aggr_net.output().data(), value_net.input().data, aggr_net.output().size());
      
      actor_net.forward();
      //value_net.forward(); //not for imitation
      auto whole_action_vec = actor_net.output().to_vector();
      auto action_vec = vector<float>(whole_action_vec.begin() + t * act_dim, whole_action_vec.begin() + (t+1) * act_dim);
      cout << "act vec: " << action_vec << endl;
      */
      
      vector<float> action_vec(act_dim);
      copy_gpu_to_cpu(aggr_net.output().data(t), &action_vec[0], act_dim);
      //action_vec[7] += .4;
      //action_vec[8] += .3;
      //action_vec[9] += .5;
      Action act(action_vec);
      //cout << action_vec << endl;
      cout << act << endl;
      //cout << "pred/actual action: " << endl << act << endl << action << endl;
      //cout << "cur/next pose: " << endl << cur_pose << endl << next_pose << endl;
      cout << "cur applied pose: " << endl << cur_pose << endl;
      cur_pose.from_scene(scene);
      cout << "after fromscene new: " << endl << cur_pose << endl;
      cur_pose.apply(act);
      cout << "new applied pose: " << endl << cur_pose << endl;
      cur_pose.apply_to_scene(scene);
      cur_pose.from_scene(scene);
      cout << "after apply new: " << endl << cur_pose << endl;
      /*int layer(0);
      for (auto v : vis_net.volumes) {
        ostringstream oss;
        oss << "images/" << t << "_layer_" << layer << ".png";
        v->x.draw_slice(oss.str(), t, 1);
        ++layer;
        }*/

      //cout << "act: " << act.armq[0] << " " << act.arm_length << endl;
      //cout << cur_pose << endl;
      
      //cur_pose.apply(act);
      //cur_pose.apply(action);
      //cout << cur_pose << endl;
      //cur_pose.apply_to_scene(scene);
    }
    //return -1;
    ++epoch;
  }
  
  Global::shutdown();
  return 0;
}

int rollout_old(string filename) {
  Global::inst().INVERT_CORRECTION = true;
  cerr << "Warning, INvert correction on!" << endl;
  return rollout(filename);
}

int fix(string filename) {
  if (!exists(filename))
    throw StringException("file doesnt exist");

  Global::inst().HEADLESS = true;

  auto &scene = Global::scene();
  Script world_script;
  world_script.run("fittsworld.lua");
  //FittsWorld world(scene);
  
  Timer a_timer(1./90);
  uint i(0);
  Recording new_recording;

  Recording recording;
  recording.load(filename, &scene);
  cout << "recording size: " << recording.size() << endl;

  
  Orientation last_orientation = Null;
  float last_xdir = 0;
  float last_ydir = 0;
  float last_zdir = 0;
  int last_choice = -1;
  
  while (i < recording.size()) {
    //cout << i << endl;
    //vr.update_track_pose();
    //scene.step();
    //scene.snap(&recording);

    
    recording.load_scene(i, &scene);

    auto substractnorm = [](float val, float c) -> float {
      if (c > val)
        return 0;
      if (val > 0)
        return val - c;
      else
        return val + c;
    };
    
    try {      
      Pos pos_box1 = scene.find<Box>("box1").p;
      Pos pos_box2 = scene.find<Box>("box2").p;
      Pos pos_box3 = scene.find<Box>("box3").p;
      
      auto w = scene.find<Box>("box1").width;
      auto h = scene.find<Box>("box1").height;
      auto d = scene.find<Box>("box1").depth;

      //determine orientation, width, xdir, ydir, zdir, start and end
      float xdir = substractnorm(pos_box2.x - pos_box1.x, w);
      float ydir = substractnorm(pos_box2.y - pos_box1.y, h);
      float zdir = substractnorm(pos_box2.z - pos_box1.z, d);

      //store experiment vars;

      Orientation orientation = Horizontal;
      float short_side = 0;
      
      if (w > h) {
        orientation = Horizontal;
        short_side = h;
      }
      if (h > d) {
        orientation = Vertical;
        short_side = d;
      }
      if (d > w) {
        orientation = Laying;
        short_side = w;
      }

      int choice = 0;
      if (scene.find<Box>("box1").tex_name == "blue-checker.png")
        choice = 0;
      if (scene.find<Box>("box2").tex_name == "blue-checker.png")
        choice = 1;
      if (scene.find<Box>("box3").tex_name == "blue-checker.png")
        choice = 2;
      //cout << i << " " << choice << " " << orientation << " " << xdir << endl;
      scene.variable<FreeVariable>("orientation").set_value(orientation);
      
      scene.variable<FreeVariable>("xdir").set_value(xdir);
      scene.variable<FreeVariable>("ydir").set_value(ydir);
      scene.variable<FreeVariable>("zdir").set_value(zdir);

      scene.variable<FreeVariable>("short_side").set_value(short_side);
      scene.variable<FreeVariable>("choice").set_value(choice);
      //scene.variable<MarkVariable>("start").set_value(1);

      if (last_choice != choice ||
               last_xdir != xdir ||
               last_ydir != ydir ||
               last_zdir != zdir ||
               last_orientation != orientation) {
        last_choice = choice;
        last_xdir = xdir;
        last_ydir = ydir;
        last_zdir = zdir;
        last_orientation = orientation;
        scene.variable("start").set_value(1);
        cout << "set start " << i << endl;
      }
    } catch(...) {
      cout << "catch " << i << endl;
      cout << last_xdir << " " << last_ydir << " " << last_zdir << " " << last_choice << " " << last_orientation << endl;
      scene.variable("end").set_value(1);
    }
    scene.snap(&new_recording);
    scene.update_variables();
    
    //cout << "var size: " << new_recording.variables.size() << endl;
    ++i;
  }
  
  new_recording.save(filename + ".fix", scene);
  
  Global::shutdown();  
  return 0;
}

int analyse(string filename) {
  if (!exists(filename))
    throw StringException("file doesnt exist");

  Global::inst().HEADLESS = true;

  auto &scene = Global::scene();
  Script world_script;
  world_script.run("fittsworld.lua");
  //FittsWorld world(scene);
  
  Timer a_timer(1./90);
  uint i(0);
  Recording recording;
  recording.load(filename, &scene);
  cout << "recording size: " << recording.size() << endl;
  /*
  for (auto o : scene.objects)
    cout << o.first << " " << scene.names[o.second->nameid] << endl;

  for (auto v : scene.variables)
    cout << v.first << " " << v.second->val << " " << scene.names[v.second->nameid] << endl;
  for (auto t : scene.triggers)
    cout << scene.names[t->function_nameid] << endl;
  */
  
  int clicked(0);
  Pos start_pos;
  int start_frame(0);
  int start_clicks(0);

  ofstream datafile(filename + ".data");
  datafile << "ORIENTATION WIDTH XDIR YDIR ZDIR STARTX STARTY STARTZ ENDX ENDY ENDZ NFRAMES" << endl;

  while (i < recording.size()) {
    //cout << i << endl;
    //vr.update_track_pose();
    //scene.step();
    //scene.snap(&recording);

    
    recording.load_scene(i, &scene);
    
    try {
      if (scene.find<Controller>("controller").clicked)
        clicked++;

      if (scene.variable<MarkVariable>("start").val > 0) {
        start_frame = i;
        start_clicks = clicked;
        start_pos = scene.find<Controller>("controller").p;
      }
      
      if (scene.variable<MarkVariable>("end").val > 0) {
        cout << "bla" << endl;
        Pos cur_pos = scene.find<Controller>("controller").p;
        //Pos cur_pos_box = scene.find<Box>("box1").p;
        // auto w = scene.find<Box>("box1").width;
        //auto h = scene.find<Box>("box1").height;
        //auto d = scene.find<Box>("box1").depth;
        cout << "bloe" << endl;
        datafile << scene.variable<FreeVariable>("orientation").val << " "
                 << scene.variable<FreeVariable>("short_side").val << " "
                 << scene.variable<FreeVariable>("xdir").val << " "
                 << scene.variable<FreeVariable>("ydir").val << " " 
                 << scene.variable<FreeVariable>("zdir").val << " "
                 << scene.variable<FreeVariable>("choice").val << " "
                 << start_pos.x << " "
                 << start_pos.y << " "
                 << start_pos.z << " "
                 << cur_pos.x << " "
                 << cur_pos.y << " "
                 << cur_pos.z << " "
                 << (i - start_frame) << endl;
        //<< scene.reward << " "
        //<< (i - start_frame) << endl;
        //start_pos = cur_pos;
      }
    } catch(...) {
      
    }
    
    //cout << "scene " << i << " items: " << scene.objects.size() << endl;
    ++i;
    //vr.request_poses();
    //a_timer.wait();
  }
  datafile.flush();
  cout << "n clicks: " << clicked << endl;
  
  Global::shutdown();
  return 0;
}

int main(int argc, char **argv) {
  vector<string> args;
  for (int i(1); i < argc; ++i)
    args.push_back(argv[i]);

  if (args.size() < 1)
    throw StringException("not enough args, use: record|replay filename");
  if (args[0] == "autoencoder") //with invert correction
    return train_autoencoder();
  if (args.size() < 2)
    throw StringException("not enough args, use: record|replay filename");
  if (args[0] == "record")
    return record(args[1]);
  if (args[0] == "replay")
    return replay(args[1]);
  if (args[0] == "analyse")
    return analyse(args[1]);
  if (args[0] == "fix")
    return fix(args[1]);
  if (args[0] == "learn")
    return learn(args[1]);
  if (args[0] == "learnold") //with invert correction
    return learn_old(args[1]);
  if (args[0] == "rollout")
    return rollout(args[1]);
  if (args[0] == "rolloutold") //with invert correction
    return rollout_old(args[1]);
  if (args[0] == "convert") {
    
    return convert(args[1]);

  }
  throw StringException("Call right arguments");

}
