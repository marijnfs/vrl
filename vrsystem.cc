#include <iostream>
#include <vector>
#include <string>

#include "img.h"
#include "vrsystem.h"
#include "util.h"
#include "utilvr.h"
#include "global.h"
#include "vulkansystem.h"
#include "shared/Matrices.h"
#include "scene.h"



using namespace std;

std::vector<float> TrackedController::get_pos() {
  return std::vector<float>{t[12], t[13], t[14]};
}

void TrackedController::set_t(Matrix4 &t_) {
  t = t_;
}

VRSystem::VRSystem() {
}

VRSystem::~VRSystem() {
  vr::VR_Shutdown();
  ivrsystem = 0;

  delete left_eye_fb;
  delete right_eye_fb;
}

void VRSystem::init() {
  if (!Global::inst().HEADLESS)
    init_full();
  else
    init_headless();
}

void VRSystem::init_headless() {
  render_width = load_val<uint32_t>("/home/marijnfs/data/vrparams/renderwidth");
  render_height = load_val<uint32_t>("/home/marijnfs/data/vrparams/renderheight");

  auto epl_vec = load_vec<float>("/home/marijnfs/data/vrparams/eyeposleft");
  auto epr_vec = load_vec<float>("/home/marijnfs/data/vrparams/eyeposright");
  
  auto pl_vec = load_vec<float>("/home/marijnfs/data/vrparams/projectionleft");
  auto pr_vec = load_vec<float>("/home/marijnfs/data/vrparams/projectionright");

  eye_pos_left.set(&epl_vec[0]);
  eye_pos_right.set(&epr_vec[0]);
  projection_left.set(&pl_vec[0]);
  projection_right.set(&pr_vec[0]);
}

void VRSystem::init_full() {
	cout << "initialising VRSystem" << endl;

	render_width = 0;
	render_height = 0;

	near_clip = 0.1f;
	far_clip = 30.0f;

	vr::EVRInitError err = vr::VRInitError_None;
	ivrsystem = vr::VR_Init( &err, vr::VRApplication_Scene );
	check(err);

    ivrsystem->GetRecommendedRenderTargetSize( &render_width, &render_height );

	render_models = (vr::IVRRenderModels *)vr::VR_GetGenericInterface( vr::IVRRenderModels_Version, &err );
    check(err);
    
	driver_str = query_str(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
	display_str = query_str(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);

	cout << "driver: " << driver_str << " display: " << display_str << endl;

	if ( !vr::VRCompositor() ) {
		cerr << "Couldn't create VRCompositor" << endl;
		throw StringException("Couldn't create VRCompositor");
	}

	eye_pos_left = get_eye_transform(vr::Eye_Left);
	eye_pos_right = get_eye_transform(vr::Eye_Right);

	projection_left = get_hmd_projection(vr::Eye_Left);
	projection_right = get_hmd_projection(vr::Eye_Right);

    save_val(render_width, "/home/marijnfs/data/vrparams/renderwidth");
    save_val(render_height, "/home/marijnfs/data/vrparams/renderheight");

    vector<float> epl_vec(eye_pos_left.get(), eye_pos_left.get() + 16);
    vector<float> epr_vec(eye_pos_right.get(), eye_pos_right.get() + 16);
    vector<float> pl_vec(projection_left.get(), projection_left.get() + 16);
    vector<float> pr_vec(projection_right.get(), projection_right.get() + 16);
    
    save_vec(epl_vec, "/home/marijnfs/data/vrparams/eyeposleft");
    save_vec(epr_vec, "/home/marijnfs/data/vrparams/eyeposright");

    save_vec(pl_vec, "/home/marijnfs/data/vrparams/projectionleft");
    save_vec(pr_vec, "/home/marijnfs/data/vrparams/projectionright");
        
	cout << "done initialising VRSystem" << endl;
}

void VRSystem::setup() {
  setup_render_targets();
  //setup_render_models();

  //Global::vk().end_submit_cmd();
}

void VRSystem::setup_render_targets() {
  cout << "recommended target size: " << render_width << "x" << render_height << endl;
  
  left_eye_fb = new FrameRenderBuffer();
  right_eye_fb = new FrameRenderBuffer();
  
  left_eye_fb->init(render_width, render_height);
  right_eye_fb->init(render_width, render_height);
}

void VRSystem::copy_image_to_cpu(std::vector<float> &img) {
  //cout << "copy image to cpu" << endl;
  
  auto &data_left = left_eye_fb->copy_to_buffer();
  auto &data_right = right_eye_fb->copy_to_buffer();
  
  //cout << "sizes: " << data_left.size() << " " << data_right.size() << endl;
  int height(left_eye_fb->height);
  int width(left_eye_fb->width);
  //cout << height << " " << width << endl;

  //write_img("test2.png", 3, width, height, &data_left[0]);
  //vector<float> img2(data_left.size());
  //for (int x = 0; x < width * height; ++x)
  //for (int c = 0; c < 3; ++c)
  //  img2[c * width * height + x] = data_left[x * 4 + c];
  //for (int x = 0; x < width * height; ++x)
    //for (int c = 0; c < 3; ++c)
      //  img2[x * 3 + c] = data_left[x * 4 + c];
  // write_img("test3.png", 3, width, height, &img2[0]);
  //throw "";
  for (int y(0); y < height; ++y)
    for (int x(0); x < width; ++x)
      for (int c(0); c < 3; ++c)
        //img[c * weight * height + y * width + x] = data_left[(y * width + x) * 4 + c];
        img[c * width * height + y * width + x] = data_left[(y * width + x) * 4 + c];
  //write_img1c("test4.png", width, height, &img[0]);
    for (int y(0); y < height; ++y)
      for (int x(0); x < width; ++x)
        for (int c(0); c < 3; ++c)
          img[((c+3) * height + y) * width + x] = data_right[(y * width + x) * 4 + c];
    //cout << "before; " <<  width << " " << height << endl;
    //write_img1c("beforereturn.png", width, height, &img[0]);
    //write_img1c("test4.png", width, height, &img[3 * height * width]);
//cout << "pixels:" << endl;
  //for (auto &v : *data_left)
  // if (v)
  //    cout << v;
  //cout << "===" << endl;
}

vector<float> VRSystem::get_image_data() {
  cout << "get img data" << endl;
  auto right_data = right_eye_fb->img.get_data<uint8_t>();
  auto left_data = left_eye_fb->img.get_data<uint8_t>();
  cout << right_data.size() << " " << left_data.size() << endl;
  throw "not implemented";
}

void VRSystem::render(Scene &scene, bool headless, std::vector<float> *img_ptr) { //needs a headless option
  auto &vk = Global::vk();

  if (!headless) {
    vk.swapchain.acquire_image(); //
    
    // RENDERING
    render_stereo_targets(scene);
    render_companion_window();
    to_present();
    vk.end_submit_swapchain_cmd();  //could try without swapchain if headless

    if (img_ptr)
      copy_image_to_cpu(*img_ptr); //later remove?
    submit_to_hmd();
    presentKHR();
  } else {
    // RENDERING
    //vk.swapchain.acquire_image(); //
    render_stereo_targets(scene);
    //render_companion_window();
    to_present();
    vk.end_submit_cmd();
    //vk.end_submit_swapchain_cmd();  //could try without swapchain if headless
    
    if (img_ptr)
      copy_image_to_cpu(*img_ptr); //later remove?
  }
}

void VRSystem::submit_to_hmd() {
  auto &vk = Global::vk();
  // Submit to SteamVR
	vr::VRTextureBounds_t bounds;
	bounds.uMin = 0.0f;
	bounds.uMax = 1.0f;
	bounds.vMin = 0.0f;
	bounds.vMax = 1.0f;

	vr::VRVulkanTextureData_t vulkanData;
    //cout << "img: " << left_eye_fb->img.img << endl;
	vulkanData.m_nImage = ( uint64_t ) left_eye_fb->img.img;
	vulkanData.m_pDevice = ( VkDevice_T * ) vk.dev;
	vulkanData.m_pPhysicalDevice = ( VkPhysicalDevice_T * ) vk.phys_dev;
	vulkanData.m_pInstance = ( VkInstance_T *) vk.inst;
	vulkanData.m_pQueue = ( VkQueue_T * ) vk.queue;
	vulkanData.m_nQueueFamilyIndex = vk.graphics_queue;

	vulkanData.m_nWidth = render_width;
	vulkanData.m_nHeight = render_height;
	vulkanData.m_nFormat = VK_FORMAT_R8G8B8A8_SRGB;
	vulkanData.m_nSampleCount = msaa;
    
    //submitting to HMD
	vr::Texture_t texture = { &vulkanData, vr::TextureType_Vulkan, vr::ColorSpace_Auto };
	vr::VRCompositor()->Submit( vr::Eye_Left, &texture, &bounds );

	vulkanData.m_nImage = ( uint64_t ) right_eye_fb->img.img;
	vr::VRCompositor()->Submit( vr::Eye_Right, &texture, &bounds );
}

void VRSystem::presentKHR() {
  auto &vk = Global::vk();

  //present (for companion window)
	VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.pNext = NULL;
	pi.swapchainCount = 1;
	pi.pSwapchains = &vk.swapchain.swapchain;
	pi.pImageIndices = &vk.swapchain.current_swapchain_image;
	vkQueuePresentKHR( vk.queue, &pi );
}

void VRSystem::render_stereo_targets(Scene &scene) {
	auto &vk = Global::vk();
	//auto &scene = Global::scene();
    hmd_pose_inverse = hmd_pose;
    hmd_pose_inverse.invert();
    
	VkViewport viewport = { 0.0f, 0.0f, (float ) render_width, ( float ) render_height, 0.0f, 1.0f };
	vkCmdSetViewport( vk.cmd_buffer(), 0, 1, &viewport );
	VkRect2D scissor = { 0, 0, render_width, render_height};
	vkCmdSetScissor( vk.cmd_buffer(), 0, 1, &scissor );


    /*
	left_eye_fb->img.to_colour_optimal();
	if (left_eye_fb->depth_stencil.layout == VK_IMAGE_LAYOUT_UNDEFINED)
		left_eye_fb->depth_stencil.to_depth_optimal();
    */ 
    left_eye_fb->img.barrier(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (left_eye_fb->depth_stencil.layout == VK_IMAGE_LAYOUT_UNDEFINED)
      left_eye_fb->depth_stencil.barrier(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    left_eye_fb->start_render_pass();
    
    auto proj_left = get_view_projection(vr::Eye_Left);
    memcpy(&draw_visitor.mvp, &proj_left, sizeof(Matrix4));
    
    //render stuff
    draw_visitor.right = false;
    scene.visit(draw_visitor);
    
    left_eye_fb->end_render_pass();
    
    left_eye_fb->img.barrier(VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


    //Right Eye
    right_eye_fb->img.barrier(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (right_eye_fb->depth_stencil.layout == VK_IMAGE_LAYOUT_UNDEFINED)
      right_eye_fb->depth_stencil.barrier(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    right_eye_fb->start_render_pass();
    


    auto proj_right = get_view_projection(vr::Eye_Right);
    memcpy(&draw_visitor.mvp, &proj_right, sizeof(Matrix4));
	
    //render stuff
    draw_visitor.right = true;
    scene.visit(draw_visitor);
    
    
    right_eye_fb->end_render_pass();
    right_eye_fb->img.barrier(VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VRSystem::render_companion_window() {
  auto &ws = Global::ws();
    auto &vk = Global::vk();
    
    auto &sc = vk.swapchain;
    
    //auto &swap_img = sc.images[sc.current_swapchain_image];
    
    //auto &fb = sc.framebuffers[sc.current_swapchain_image];



    // Start the renderpass
    //sc.to_colour_optimal(sc.current_swapchain_image);
    sc.current_img().barrier(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    sc.begin_render_pass(ws.width, ws.height);

    
    // Set viewport/scissor
    VkViewport viewport = { 0.0f, 0.0f, (float ) ws.width, ( float ) ws.height, 0.0f, 1.0f };
    vkCmdSetViewport( vk.cmd_buffer(), 0, 1, &viewport );
    VkRect2D scissor = { 0, 0, ws.width, ws.height };
    vkCmdSetScissor( vk.cmd_buffer(), 0, 1, &scissor );
    
    // Bind the pipeline and descriptor set
    vkCmdBindPipeline( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelines[ PSO_COMPANION ] );
    vkCmdBindDescriptorSets( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline_layout, 0, 1, &left_eye_fb->desc.desc, 0, nullptr );

    // Draw left eye texture to companion window
    VkDeviceSize nOffsets[ 1 ] = { 0 };
    vkCmdBindVertexBuffers( vk.cmd_buffer(), 0, 1, &ws.vertex_buf.buffer, &nOffsets[ 0 ] );
    vkCmdBindIndexBuffer( vk.cmd_buffer(), ws.index_buf.buffer, 0, VK_INDEX_TYPE_UINT16 );
    vkCmdDrawIndexed( vk.cmd_buffer(), ws.index_buf.size() / 2, 1, 0, 0, 0 );
    
    // Draw right eye texture to companion window
    vkCmdBindDescriptorSets( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline_layout, 0, 1, &right_eye_fb->desc.desc, 0, nullptr );
    vkCmdDrawIndexed( vk.cmd_buffer(), ws.index_buf.size() / 2, 1, ws.index_buf.size() / 2, 0, 0 );
    
    // End the renderpass
    sc.end_render_pass();

    //sc.to_present_optimal(sc.current_swapchain_image);
    //left_eye_fb->img.to_transfer_src();
    //right_eye_fb->img.to_transfer_src();
}

void VRSystem::to_present() {
  if (!Global::inst().HEADLESS)
    Global::vk().swapchain.to_present();
  left_eye_fb->img.barrier(VK_ACCESS_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  right_eye_fb->img.barrier(VK_ACCESS_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

Matrix4 VRSystem::get_eye_transform( vr::Hmd_Eye eye )
{
	vr::HmdMatrix34_t mat_eye = ivrsystem->GetEyeToHeadTransform( eye );
	Matrix4 mat(
		mat_eye.m[0][0], mat_eye.m[1][0], mat_eye.m[2][0], 0.0, 
		mat_eye.m[0][1], mat_eye.m[1][1], mat_eye.m[2][1], 0.0,
		mat_eye.m[0][2], mat_eye.m[1][2], mat_eye.m[2][2], 0.0,
		mat_eye.m[0][3], mat_eye.m[1][3], mat_eye.m[2][3], 1.0f
		);

	return mat.invert();
}

Matrix4 VRSystem::get_hmd_projection( vr::Hmd_Eye eye )
{
	vr::HmdMatrix44_t mat = ivrsystem->GetProjectionMatrix( eye, near_clip, far_clip );

	return Matrix4(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
		mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1], 
		mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2], 
		mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
		);
}


Matrix4 VRSystem::get_view_projection( vr::Hmd_Eye eye ) {
	if( eye == vr::Eye_Left )
		return projection_left * eye_pos_left * hmd_pose_inverse;
	else if( eye == vr::Eye_Right )
		return projection_right * eye_pos_right * hmd_pose_inverse;
	throw StringException("not valid eye");
}


void VRSystem::setup_render_models()
{
  for( uint32_t d = vr::k_unTrackedDeviceIndex_Hmd + 1; d < vr::k_unMaxTrackedDeviceCount; d++ )
	{
		if( !ivrsystem->IsTrackedDeviceConnected( d ) )
			continue;

		//TODO: Setup render model

		//SetupRenderModelForTrackedDevice( d );
	}
}

void VRSystem::setup_render_model_for_device(int d) {
  //todo, create graphics object or something for controllers
}

void VRSystem::request_poses() {
  //TrackingUniverseSeated
  //TrackingUniverseRawAndUncalibrated
  // for somebody asking for the default figure out the time from now to photons.
  float last_vsync(0);
  ivrsystem->GetTimeSinceLastVsync( &last_vsync, NULL );

  float freq = ivrsystem->GetFloatTrackedDeviceProperty( vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float );
  float dur = 1.f / freq;
  float sync_to_photons = ivrsystem->GetFloatTrackedDeviceProperty( vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SecondsFromVsyncToPhotons_Float );

  float prediction = dur - last_vsync + sync_to_photons;
  
  ivrsystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseSeated, prediction, tracked_pose, vr::k_unMaxTrackedDeviceCount);
}

void VRSystem::wait_frame() {
  vr::VRCompositor()->WaitGetPoses(tracked_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0 );
}

void VRSystem::update_track_pose() {
  //vr::VRControllerState_t cstate;
  //ivrsystem->GetControllerState(ivrsystem->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_RightHand), &cstate, sizeof(cstate));
  //cout << "axis 1: " << cstate.rAxis[1].x << endl;
                                
//
  
  int controller_idx(0);
  //cout << "updating track pose " << endl;

  //vr::VRCompositor()->WaitGetPoses(tracked_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0 );
  vr::VRCompositor()->GetLastPoses(tracked_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0 );
  
  //cout << "done" << endl;
	for ( int d = 0; d < vr::k_unMaxTrackedDeviceCount; ++d) {
      if ( tracked_pose[d].bPoseIsValid ) {
        //cout << "updating: " << d << endl;
        tracked_pose_mat4[d] = vrmat_to_mat4( tracked_pose[d].mDeviceToAbsoluteTracking );
        device_class[d] = ivrsystem->GetTrackedDeviceClass(d);
        //todo
        
        
        if (device_class[d] == vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
          vr::VRControllerState_t cstate;
          //ivrsystem->GetControllerState((vr::TrackedDeviceIndex_t)d, &cstate, sizeof(vr::TrackedDeviceIndex_t));
          if (!ivrsystem->GetControllerState((vr::TrackedDeviceIndex_t)d, &cstate, sizeof(cstate)))
            throw "no info";
          
          //cout << "controller buttons" << cstate.ulButtonPressed << endl;
          if (controller_idx == 0) {
            right_controller.set_t(tracked_pose_mat4[d]);
            //cout << "AXIS: " << cstate.rAxis[1].x << endl;
            right_controller.pressed = cstate.rAxis[1].x > .4;
          } else {
            left_controller.set_t(tracked_pose_mat4[d]);
            left_controller.pressed = cstate.rAxis[1].x > .4;
          }
          ++controller_idx;
        }
      }
	}
    
	if ( tracked_pose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid ) {
      hmd_pose = tracked_pose_mat4[vr::k_unTrackedDeviceIndex_Hmd];
    }
}

string VRSystem::query_str(vr::TrackedDeviceIndex_t devidx, vr::TrackedDeviceProperty prop) {
	vr::TrackedPropertyError *err = NULL;
	uint32_t buflen = ivrsystem->GetStringTrackedDeviceProperty( devidx, prop, NULL, 0, err );
	if( buflen == 0)
		return "";

	string buf(buflen, ' ');
	buflen = ivrsystem->GetStringTrackedDeviceProperty( devidx, prop, &buf[0], buflen, err );
	return buf;      
}

vector<string> VRSystem::get_inst_ext_required() {
  if (Global::inst().HEADLESS)
    return load_vec<string>("/home/marijnfs/data/vrparams/get_inst_ext_required");
  uint32_t buf_size = vr::VRCompositor()->GetVulkanInstanceExtensionsRequired( nullptr, 0 );
  if (!buf_size)
    throw StringException("no such GetVulkanInstanceExtensionsRequired");
  
  string buf(buf_size, ' ');
  vr::VRCompositor()->GetVulkanInstanceExtensionsRequired( &buf[0], buf_size );
  cout << "ext required: " << buf << endl;
  // Break up the space separated list into entries on the CUtlStringList
  vector<string> ext_list;
  string cur_ext;
  uint32_t idx = 0;
  while ( idx < buf_size ) {
    if ( buf[ idx ] == ' ' ) {
      ext_list.push_back( cur_ext );
      cur_ext.clear();
    } else {
      cur_ext += buf[ idx ];
    }
    ++idx;
  }
  if ( cur_ext.size() > 0 ) {
    ext_list.push_back( cur_ext );
  }
  
  save_vec(ext_list, "/home/marijnfs/data/vrparams/get_inst_ext_required");
  return ext_list;
}

vector<string> VRSystem::get_dev_ext_required() {
  if (Global::inst().HEADLESS)
    return load_vec<string>("/home/marijnfs/data/vrparams/get_dev_ext_required");

  auto &vk = Global::vk();
    cout << "phys dev ptr:" << vk.phys_dev << endl;
	uint32_t buf_size = vr::VRCompositor()->GetVulkanDeviceExtensionsRequired( vk.phys_dev, nullptr, 0 );
	if (!buf_size)
		throw StringException("No such GetVulkanDeviceExtensionsRequired");

	string buf(buf_size, ' ');
	vr::VRCompositor()->GetVulkanDeviceExtensionsRequired( vk.phys_dev, &buf[0], buf_size );

    cout << buf << endl;
    // Break up the space separated list into entries on the CUtlStringList
	vector<string> ext_list;
	string cur_ext;
	uint32_t idx = 0;
	while ( idx < buf_size ) {
      if (buf[idx] == 0)
        break;
      if ( buf[ idx ] == ' ' ) {
        ext_list.push_back( cur_ext );
        cur_ext.clear();
      } else {
        cur_ext += buf[ idx ];
      }
      ++idx;
	}
	if ( cur_ext.size() > 0 ) {
      ext_list.push_back( cur_ext );
	}
  save_vec(ext_list, "/home/marijnfs/data/vrparams/get_dev_ext_required");
  return ext_list;
}

vector<string> VRSystem::get_inst_ext_required_verified() {
	auto instance_ext_req = get_inst_ext_required();
    cout << "extension size:" << instance_ext_req.size() << endl;
    for (auto ext : instance_ext_req)
      cout << ext << endl;
	instance_ext_req.push_back( VK_KHR_SURFACE_EXTENSION_NAME );

#if defined ( _WIN32 )
	instance_ext_req.push_back( VK_KHR_WIN32_SURFACE_EXTENSION_NAME );
#else
	instance_ext_req.push_back( VK_KHR_XLIB_SURFACE_EXTENSION_NAME );
#endif

    //todo remove return	
	return instance_ext_req;
    
	uint32_t n_instance_ext(0);
	check( vkEnumerateInstanceExtensionProperties( NULL, &n_instance_ext, NULL ), "vkEnumerateInstanceExtensionProperties");

	vector<VkExtensionProperties> ext_prop(n_instance_ext);

	check( vkEnumerateInstanceExtensionProperties( NULL, &n_instance_ext, &ext_prop[0]), "vkEnumerateInstanceExtensionProperties" );

	for (auto req_inst : instance_ext_req) {
      cout << ":" << req_inst << endl;
		bool found(false);
		for (auto prop : ext_prop) 
			if (req_inst == string(prop.extensionName))
				found = true;
		if (!found) {
          cerr << "couldn't find extension " << req_inst << endl;
			throw "";
		}
	}
		
	return instance_ext_req;
}

vector<string> VRSystem::get_dev_ext_required_verified() {
	auto &vk = Global::vk();
	auto dev_ext_req = get_dev_ext_required();

	uint32_t n_dev_ext(0);
	check( vkEnumerateDeviceExtensionProperties( vk.phys_dev, NULL, &n_dev_ext, NULL ), "vkEnumerateDeviceExtensionProperties");

	vector<VkExtensionProperties> ext_prop(n_dev_ext);

	check( vkEnumerateDeviceExtensionProperties( vk.phys_dev, NULL, &n_dev_ext, &ext_prop[0]), "vkEnumerateDeviceExtensionProperties" );

    int n(0);
    
	for (auto req_dev : dev_ext_req) {
		bool found(false);
		for (auto prop : ext_prop)
          if (req_dev == string(prop.extensionName))
            found = true;
        
        if (!found) {
          cerr << "couldn't find extension: :" << req_dev << ":" << endl;
          throw "";
        }
	}
		
	return dev_ext_req;
}

uint64_t VRSystem::get_output_device(VkInstance v_inst) {
  uint64_t hmd_dev(0);
  ivrsystem->GetOutputDevice(&hmd_dev, vr::TextureType_Vulkan, v_inst);
  return hmd_dev;
}


