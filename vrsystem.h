#ifndef __VRSYSTEM_H__
#define __VRSYSTEM_H__

#include <vector>
#include <string>
#include <openvr.h>

#include "shared/Matrices.h"
#include "buffer.h"
#include "vulkansystem.h"
#include "scene.h"
#include "framerenderbuffer.h"
#include <vulkan/vulkan.h>
struct TrackedController {
  Matrix4 t;
  bool pressed = false;
  
    std::vector<float> get_pos();
  void set_t(Matrix4 &t);
};


struct VRSystem {
  vr::IVRSystem *ivrsystem = 0;
  vr::IVRRenderModels *render_models = 0;
  std::string driver_str, display_str;

  //tracking vars

  vr::TrackedDevicePose_t tracked_pose[ vr::k_unMaxTrackedDeviceCount ];
  Matrix4 tracked_pose_mat4[ vr::k_unMaxTrackedDeviceCount ];
  vr::TrackedDeviceClass device_class[ vr::k_unMaxTrackedDeviceCount ];

  //common matrices
  Matrix4 hmd_pose, hmd_pose_inverse;
  Matrix4 eye_pos_left, eye_pos_right, eye_pose_center;
  Matrix4 projection_left, projection_right;

  //controllers;
  TrackedController left_controller, right_controller;
  
  //render targets
  FrameRenderBuffer *left_eye_fb = 0, *right_eye_fb = 0;

  //Buffer left_eye_buf, right_eye_buf;
  //void *left_eye_mvp, *right_eye_mvp;
  
  ////buffers
  //std::vector<Buffer> eye_pos_buffer;

  uint32_t render_width = 0, render_height = 0;
  float near_clip = 0, far_clip = 0;

  DrawVisitor draw_visitor; //visitor pattern to draw the scene
  Image dst_image_left, dst_image_right;
  
  VRSystem();
  ~VRSystem();

  void init();
  void init_headless();
  void init_full();
  void setup();
  
  Matrix4 get_eye_transform( vr::Hmd_Eye eye );
  Matrix4 get_hmd_projection( vr::Hmd_Eye eye );
  Matrix4 get_view_projection( vr::Hmd_Eye eye );

  void request_poses();
  void update_track_pose();
  void wait_frame();
  
  void render(Scene &scene, bool headless = false, std::vector<float> *img_ptr = 0);
  void render_stereo_targets(Scene &scene);
  void render_companion_window();

  void copy_image_to_cpu(std::vector<float> &img);
  std::vector<float> get_image_data();

  void submit_to_hmd();
  void presentKHR();
  void to_present();
  
  void setup_render_models();
  void setup_render_model_for_device(int d);
  void setup_render_targets();
  uint64_t get_output_device(VkInstance v_inst);

  std::string query_str(vr::TrackedDeviceIndex_t devidx, vr::TrackedDeviceProperty prop);

  std::vector<std::string> get_inst_ext_required();
  std::vector<std::string> get_dev_ext_required();
  std::vector<std::string> get_inst_ext_required_verified();
  std::vector<std::string> get_dev_ext_required_verified();

  };

#endif
