#include "vulkansystem.h"
#include "global.h"
#include "util.h"
#include "utilvr.h"
#include "shared/lodepng.h"

using namespace std;

FencedCommandBuffer::FencedCommandBuffer() {
  init();
}

//  ==== FENCED BUFFER ====
void FencedCommandBuffer::begin() {
  VkCommandBufferBeginInfo cmbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  cmbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer( cmd_buffer, &cmbbi );
}

void FencedCommandBuffer::end() {
  vkEndCommandBuffer( cmd_buffer );
}

bool FencedCommandBuffer::ready() {
  return vkGetFenceStatus( Global::vk().dev, fence ) == VK_SUCCESS;	
}

void FencedCommandBuffer::reset() {
  vkResetCommandBuffer( cmd_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT );
  vkResetFences( Global::vk().dev, 1, &fence );
}

void FencedCommandBuffer::init() {
  auto &vk = Global::vk();
   VkCommandBufferAllocateInfo cmd_buffer_alloc_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cmd_buffer_alloc_info.commandBufferCount = 1;
  cmd_buffer_alloc_info.commandPool = vk.cmd_pool;
  cmd_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  vkAllocateCommandBuffers( vk.dev, &cmd_buffer_alloc_info, &cmd_buffer );

  VkFenceCreateInfo fenceci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  vkCreateFence( vk.dev, &fenceci, nullptr, &fence );
}

// ==== Render Model ====
GraphicsObject::GraphicsObject() {
  
}

GraphicsObject::~GraphicsObject() {
  if (mvp_left) {
    mvp_buffer_left.unmap();
    //delete mvp_left;
  }
  if (mvp_right) {
    mvp_buffer_right.unmap();
    //delete mvp_right;
  }
  mvp_left = 0;
  mvp_right = 0;
}

void GraphicsObject::init_buffers() {
  n_index = indices.size();
  
  if (n_vertex)
    vertex_buf.init(vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, HOST);
  if (n_index)
    index_buf.init(indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, HOST);
  
  //mvp_left = new Matrix4();
  //mvp_right = new Matrix4();

  mvp_buffer_left.init(sizeof(Matrix4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HOST_COHERENT);
  mvp_buffer_right.init(sizeof(Matrix4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HOST_COHERENT);
  
  mvp_buffer_left.map((void**)&mvp_left);
  mvp_buffer_right.map((void**)&mvp_right);

  mvp_left->identity();
  mvp_right->identity();

  //image.init_from_img("", VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}


// ==== Graphics Object ====
void GraphicsObject::render(Matrix4 &mvp, bool right) {

  //TODO fix
  auto &vk = Global::vk();
  vkCmdBindPipeline( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelines[ PSO_SCENE ] );

  // Update the persistently mapped pointer to the CB data with the latest matrix, TODO: SET THIS SOMEWHERE
  //TODO set eye matrix

  if (right)
    memcpy(&mvp_right->m, &mvp.m[0], sizeof(Matrix4));
  else
    memcpy(&mvp_left->m, &mvp.m[0], sizeof(Matrix4));
  
  if (right)
    vkCmdBindDescriptorSets( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline_layout, 0, 1, &desc_right.desc, 0, nullptr );
  else
    vkCmdBindDescriptorSets( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline_layout, 0, 1, &desc_left.desc, 0, nullptr );

  // Draw
  VkDeviceSize offsets[ 1 ] = { 0 };
 
  vkCmdBindVertexBuffers( vk.cmd_buffer(), 0, 1, &vertex_buf.buffer, &offsets[ 0 ] );
  if (!index_buf.size())
    vkCmdDraw( vk.cmd_buffer(), n_vertex, 1, 0, 0 );
  else {
    vkCmdBindIndexBuffer( vk.cmd_buffer(), index_buf.buffer, 0, VK_INDEX_TYPE_UINT16 );
    vkCmdDrawIndexed( vk.cmd_buffer(), n_vertex, 1, 0, 0, 0 );
  }
}

void GraphicsObject::add_vertex(float x, float y, float z, float tx, float ty) {
  vertices.push_back( x );
  vertices.push_back( y );
  vertices.push_back( z );
  vertices.push_back( tx );
  vertices.push_back( ty );
  n_vertex++;
}


GraphicsCanvas::GraphicsCanvas() : texture("stub.png") {
  init();
}

GraphicsCanvas::GraphicsCanvas(string tex_) : texture(tex_) {
  init();
}

void GraphicsCanvas::init() {
  cout << "init Graphics Canvas" << endl;
  add_vertex( 0, 0, 1, 0, 1); //Front
  add_vertex( 1, 0, 1, 1, 1);
  add_vertex( 1, 1, 1, 1, 0);
  add_vertex( 1, 1, 1, 1, 0);
  add_vertex( 0, 1, 1, 0, 0);
  add_vertex( 0, 0, 1, 0, 1);

  init_buffers();
  auto *img = ImageFlywheel::image(texture);
  
  desc_left.register_model_texture(mvp_buffer_left.buffer, img->view, img->sampler);
  desc_right.register_model_texture(mvp_buffer_right.buffer, img->view, img->sampler);
}

GraphicsCube::GraphicsCube() : texture("stub.png") {
  init();
}

void GraphicsCube::init() {
  set_vertices();
  
  init_buffers();
  auto *img = ImageFlywheel::image(texture);
  
  desc_left.register_model_texture(mvp_buffer_left.buffer, img->view, img->sampler);
  desc_right.register_model_texture(mvp_buffer_right.buffer, img->view, img->sampler);
}

void GraphicsCube::change_dim(float width_, float height_, float depth_) {
  if (width == width_ && height == height_ && depth == depth_)
    return;
  
  width = width_;
  height = height_;
  depth = depth_;

  set_vertices();
  vertex_buf.update(vertices);
}

void GraphicsCube::set_vertices() {
  vertices.clear();
  n_vertex = 0;

  float left = 0;
  float right = width;
  float bottom = 0;
  float top = height;
  float near = 0;
  float far = depth;
  
  if (balanced) {
    left = -width / 2.0;
    right = width / 2.0;
    bottom = -height / 2.0;
    top = height / 2.0;
    near = -depth / 2.0;
    far = depth / 2.0;
  }


  Vector4 A = Vector4( left, bottom, near, 1 );
  Vector4 B = Vector4( right, bottom, near, 1 );
  Vector4 C = Vector4( right, top, near, 1 );
  Vector4 D = Vector4( left, top, near, 1 );
  Vector4 E = Vector4( left, bottom, far, 1 );
  Vector4 F = Vector4( right, bottom, far, 1 );
  Vector4 G = Vector4( right, top, far, 1 );
  Vector4 H = Vector4( left, top, far, 1 );

  // triangles instead of quads
  add_vertex( E.x, E.y, E.z, 0, 1); //Front
  add_vertex( F.x, F.y, F.z, 1, 1);
  add_vertex( G.x, G.y, G.z, 1, 0);
  add_vertex( G.x, G.y, G.z, 1, 0);
  add_vertex( H.x, H.y, H.z, 0, 0);
  add_vertex( E.x, E.y, E.z, 0, 1);

  add_vertex( B.x, B.y, B.z, 0, 1); //Back
  add_vertex( A.x, A.y, A.z, 1, 1);
  add_vertex( D.x, D.y, D.z, 1, 0);
  add_vertex( D.x, D.y, D.z, 1, 0);
  add_vertex( C.x, C.y, C.z, 0, 0);
  add_vertex( B.x, B.y, B.z, 0, 1);

  add_vertex( H.x, H.y, H.z, 0, 1); //Top
  add_vertex( G.x, G.y, G.z, 1, 1);
  add_vertex( C.x, C.y, C.z, 1, 0);
  add_vertex( C.x, C.y, C.z, 1, 0);
  add_vertex( D.x, D.y, D.z, 0, 0);
  add_vertex( H.x, H.y, H.z, 0, 1);

  add_vertex( A.x, A.y, A.z, 0, 1); //Bottom
  add_vertex( B.x, B.y, B.z, 1, 1);
  add_vertex( F.x, F.y, F.z, 1, 0);
  add_vertex( F.x, F.y, F.z, 1, 0);
  add_vertex( E.x, E.y, E.z, 0, 0);
  add_vertex( A.x, A.y, A.z, 0, 1);

  add_vertex( A.x, A.y, A.z, 0, 1); //Left
  add_vertex( E.x, E.y, E.z, 1, 1);
  add_vertex( H.x, H.y, H.z, 1, 0);
  add_vertex( H.x, H.y, H.z, 1, 0);
  add_vertex( D.x, D.y, D.z, 0, 0);
  add_vertex( A.x, A.y, A.z, 0, 1);

  add_vertex( F.x, F.y, F.z, 0, 1); //Right
  add_vertex( B.x, B.y, B.z, 1, 1);
  add_vertex( C.x, C.y, C.z, 1, 0);
  add_vertex( C.x, C.y, C.z, 1, 0);
  add_vertex( G.x, G.y, G.z, 0, 0);
  add_vertex( F.x, F.y, F.z, 0, 1);
}

// ===== SwapChain =======

void Swapchain::init() {
  cout << "initialising swapchain" << endl;

  auto &ws = Global::ws();
  auto &vk = Global::vk();

  SDL_SysWMinfo wm_info;
  SDL_VERSION( &wm_info.version );
  SDL_GetWindowWMInfo( ws.window, &wm_info );

#ifdef VK_USE_PLATFORM_WIN32_KHR
  VkWin32SurfaceCreateInfoKHR win32SurfaceCreateInfo = {};
  win32SurfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  win32SurfaceCreateInfo.pNext = NULL;
  win32SurfaceCreateInfo.flags = 0;
  win32SurfaceCreateInfo.hinstance = GetModuleHandle( NULL );
  win32SurfaceCreateInfo.hwnd = ( HWND ) wm_info.info.win.window;
  check( vkCreateWin32SurfaceKHR( inst, &win32SurfaceCreateInfo, nullptr, &surface ), "vkCreateWin32SurfaceKHR");
#else
  VkXlibSurfaceCreateInfoKHR xlib_ci = { VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR };
  xlib_ci.flags = 0;
  xlib_ci.dpy = wm_info.info.x11.display;
  xlib_ci.window = wm_info.info.x11.window;
  check( vkCreateXlibSurfaceKHR( vk.inst, &xlib_ci, nullptr, &surface ), "vkCreateXlibSurfaceKHR" );
#endif

  VkBool32 supports_present = VK_FALSE;
  check( vkGetPhysicalDeviceSurfaceSupportKHR( vk.phys_dev, vk.graphics_queue, surface, &supports_present), "vkGetPhysicalDeviceSurfaceSupportKHR");
  if (supports_present == VK_FALSE) {
    cerr << "support not present, vkGetPhysicalDeviceSurfaceSupportKHR" << endl;
    throw StringException("support not present, vkGetPhysicalDeviceSurfaceSupportKHR");
  }

  //query supported formats
  VkFormat swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  uint32_t format_index(0);
  uint32_t n_swap_format(0);
  VkColorSpaceKHR color_space;

  check( vkGetPhysicalDeviceSurfaceFormatsKHR( vk.phys_dev, surface, &n_swap_format, NULL), "vkGetPhysicalDeviceSurfaceFormatsKHR");
  vector<VkSurfaceFormatKHR> swap_formats(n_swap_format);
  check( vkGetPhysicalDeviceSurfaceFormatsKHR( vk.phys_dev, surface, &n_swap_format, &swap_formats[0]), "vkGetPhysicalDeviceSurfaceFormatsKHR");

  if (n_swap_format == 1 && swap_formats[0].format == VK_FORMAT_UNDEFINED)
    throw "strange";
  else
    for (int i(0); i < n_swap_format; ++i) {
      if (swap_formats[i].format == VK_FORMAT_B8G8R8A8_SRGB || swap_formats[i].format == VK_FORMAT_R8G8B8A8_SRGB) {
        format_index = i;
        break;
      }
    }
  swap_format = swap_formats[format_index].format;
  color_space = swap_formats[format_index].colorSpace;

  //check capabilities
  VkSurfaceCapabilitiesKHR surface_caps = {};
  check( vkGetPhysicalDeviceSurfaceCapabilitiesKHR( vk.phys_dev, surface, &surface_caps ), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

  uint32_t n_present_modes(0);
  check( vkGetPhysicalDeviceSurfacePresentModesKHR( vk.phys_dev, surface, &n_present_modes, NULL ), "vkGetPhysicalDeviceSurfacePresentModesKHR");
  vector<VkPresentModeKHR> present_modes(n_present_modes);
  check( vkGetPhysicalDeviceSurfacePresentModesKHR( vk.phys_dev, surface, &n_present_modes, &present_modes[0] ), "vkGetPhysicalDeviceSurfacePresentModesKHR");

  //create extent
  VkExtent2D swapchain_extent;
  if ( surface_caps.currentExtent.width == -1 ) {
	// If the surface size is undefined, the size is set to the size of the images requested.
    swapchain_extent.width = ws.width;
    swapchain_extent.height = ws.height;
  } else {
	// If the surface size is defined, the swap chain size must match
    swapchain_extent = surface_caps.currentExtent;
  }


  //find best present mode
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (auto mode : present_modes) {
    if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      present_mode = mode;
      break;
    }
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
      present_mode = mode;
    if (mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR && mode != VK_PRESENT_MODE_MAILBOX_KHR)
      present_mode = mode;	
  }


  n_swap = surface_caps.minImageCount;
  if (n_swap < 2) n_swap = 2;
  if (surface_caps.maxImageCount > 0 && n_swap > surface_caps.maxImageCount)
    n_swap = surface_caps.maxImageCount;

  VkSurfaceTransformFlagsKHR pre_transform;
  if (surface_caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
    pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  else
    pre_transform = surface_caps.currentTransform;

  VkImageUsageFlags image_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if ( surface_caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT )
    image_usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  else
    cerr << "Vulkan swapchain does not support VK_IMAGE_USAGE_TRANSFER_DST_BIT. Some operations may not be supported.\n" << endl;


  VkSwapchainCreateInfoKHR scci = {};
  scci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  scci.pNext = NULL;
  scci.surface = surface;
  scci.minImageCount = n_swap;
  scci.imageFormat = swap_format;
  scci.imageColorSpace = color_space;
  scci.imageExtent = swapchain_extent;
  scci.imageUsage = image_usage;
  scci.preTransform = ( VkSurfaceTransformFlagBitsKHR ) pre_transform;
  scci.imageArrayLayers = 1;
  scci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  scci.queueFamilyIndexCount = 0;
  scci.pQueueFamilyIndices = NULL;
  scci.presentMode = present_mode;
  scci.clipped = VK_TRUE;


  if (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
    scci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  else if (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
    scci.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  else
    cerr << "Unexpected value for VkSurfaceCapabilitiesKHR.compositeAlpha:" << surface_caps.supportedCompositeAlpha << endl;

  check( vkCreateSwapchainKHR( vk.dev, &scci, NULL, &swapchain), "vkCreateSwapchainKHR");

  check( vkGetSwapchainImagesKHR(vk.dev, swapchain, &n_swap, NULL), "vkGetSwapchainImagesKHR");
  vk_images.resize(n_swap);
  check( vkGetSwapchainImagesKHR(vk.dev, swapchain, &n_swap, &vk_images[0]), "vkGetSwapchainImagesKHR");
  images.resize(n_swap);
  for (int i(0); i < n_swap; ++i)
    images[i] = new Image(vk_images[i], swap_format, VK_IMAGE_ASPECT_COLOR_BIT);


  // Create a renderpass
  uint32_t n_att = 1;
  VkAttachmentDescription att_desc;
  VkAttachmentReference att_ref;
  att_ref.attachment = 0;
  att_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  att_desc.format = swap_format;
  att_desc.samples = VK_SAMPLE_COUNT_1_BIT;
  att_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  att_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att_desc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  att_desc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  att_desc.flags = 0;

  VkSubpassDescription subpassci = { };
  subpassci.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpassci.flags = 0;
  subpassci.inputAttachmentCount = 0;
  subpassci.pInputAttachments = NULL;
  subpassci.colorAttachmentCount = 1;
  subpassci.pColorAttachments = &att_ref;
  subpassci.pResolveAttachments = NULL;
  subpassci.pDepthStencilAttachment = NULL;
  subpassci.preserveAttachmentCount = 0;
  subpassci.pPreserveAttachments = NULL;

  VkRenderPassCreateInfo renderpassci = { };
  renderpassci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpassci.flags = 0;
  renderpassci.attachmentCount = 1;
  renderpassci.pAttachments = &att_desc;
  renderpassci.subpassCount = 1;
  renderpassci.pSubpasses = &subpassci;
  renderpassci.dependencyCount = 0;
  renderpassci.pDependencies = NULL;

  check( vkCreateRenderPass( vk.dev, &renderpassci, NULL, &render_pass), "vkCreateRenderPass");

  for (int i(0); i < vk_images.size(); ++i) {
    VkImageView attachments[ 1 ] = { images[i]->view };
    VkFramebufferCreateInfo fbci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fbci.renderPass = render_pass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &attachments[ 0 ];
    fbci.width = ws.width;
    fbci.height = ws.height;
    fbci.layers = 1;
    VkFramebuffer framebuffer;
    check( vkCreateFramebuffer( vk.dev, &fbci, NULL, &framebuffer ), "vkCreateFramebuffer");
    framebuffers.push_back( framebuffer );

    VkSemaphoreCreateInfo semci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    vkCreateSemaphore( vk.dev, &semci, nullptr, &semaphore );
    semaphores.push_back( semaphore );
  }

  for (auto &img : images)
    img->barrier(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
 
  
  cout << "done initialising swapchain" << endl;

}

// ==== Vulkan System====
void Swapchain::begin_render_pass(uint32_t width, uint32_t height) {
  auto &vk = Global::vk();
  VkRenderPassBeginInfo renderPassBeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
  renderPassBeginInfo.renderPass = render_pass;
  renderPassBeginInfo.framebuffer = framebuffers[current_swapchain_image];
  renderPassBeginInfo.renderArea.offset.x = 0;
  renderPassBeginInfo.renderArea.offset.y = 0;
  renderPassBeginInfo.renderArea.extent.width = width;
  renderPassBeginInfo.renderArea.extent.height = height;
  VkClearValue clearValues[ 1 ];
  clearValues[ 0 ].color.float32[ 0 ] = 0.0f;
  clearValues[ 0 ].color.float32[ 1 ] = 0.0f;
  clearValues[ 0 ].color.float32[ 2 ] = 0.0f;
  clearValues[ 0 ].color.float32[ 3 ] = 1.0f;
  renderPassBeginInfo.clearValueCount = _countof( clearValues );
  renderPassBeginInfo.pClearValues = &clearValues[ 0 ];
  vkCmdBeginRenderPass( vk.cmd_buffer(), &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE );
}

void Swapchain::end_render_pass() {
  auto &vk = Global::vk();
  vkCmdEndRenderPass( vk.cmd_buffer() );
}

void Swapchain::inc_frame() {
  frame_idx = (frame_idx + 1) % images.size();
}

void Swapchain::to_present() {
  current_img().barrier(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

VulkanSystem::VulkanSystem() {
  
}

VulkanSystem::~VulkanSystem() {
  for (auto &cmd_buf : cmd_buffers) {
    vkFreeCommandBuffers(dev, cmd_pool, 1, &cmd_buf->cmd_buffer);
    vkDestroyFence(dev, cmd_buf->fence, nullptr);
  }

  vkDestroyCommandPool(dev, cmd_pool, nullptr);
  vkDestroyDescriptorPool(dev, desc_pool, nullptr);
  
  vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(dev, desc_set_layout, nullptr);
  for (int i(0); i < PSO_COUNT; ++i) {
    vkDestroyPipeline(dev, pipelines[i], nullptr);
    vkDestroyShaderModule(dev, shader_modules_vs[i], nullptr);
    vkDestroyShaderModule(dev, shader_modules_ps[i], nullptr);
  }
  vkDestroyPipelineCache(dev, pipeline_cache, nullptr);

  if (validation)
    destroy_debug_callback();

  swapchain.destroy();
  
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
}


void VulkanSystem::submit(VkCommandBuffer cmd, VkFence fence, VkSemaphore semaphore) {
  VkSubmitInfo submiti = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
  submiti.commandBufferCount = 1;
  submiti.pCommandBuffers = &cmd;

  VkPipelineStageFlags dst_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (semaphore) {
    submiti.waitSemaphoreCount = 1;
    submiti.pWaitSemaphores = &semaphore;
    submiti.pWaitDstStageMask = &dst_mask;
  }
    
  vkQueueSubmit( queue, 1, &submiti, fence );
}

void VulkanSystem::wait_queue() {
  vkQueueWaitIdle( queue );
}

void VulkanSystem::wait_idle() {
  vkDeviceWaitIdle(dev);
}

void VulkanSystem::flush_cmd() {
  end_submit_cmd();
  wait_queue();
}

void VulkanSystem::init() {
  cout << "initialising Vulkan System" << endl;
  init_instance();
  if (validation)
    init_debug_callback();
  cout << "============" << endl;
  init_device();
  init_cmd_pool();
  if (!Global::inst().HEADLESS)
    swapchain.init();
  
  init_descriptor_sets();
}

void VulkanSystem::setup() {
  cout << "============" << endl;
  
  init_shaders();
    
  cout << "Done initialising Vulkan System" << endl;
}

void VulkanSystem::init_instance() {
  VkApplicationInfo app_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app_info.pApplicationName = "hellovr_vulkan";
  app_info.applicationVersion = 1;
  app_info.pEngineName = nullptr;
  app_info.engineVersion = 1;
  app_info.apiVersion = VK_MAKE_VERSION( 1, 0, 0 );

  vector<char*> enabled_layer_names;
    
  auto inst_req = Global::vr().get_inst_ext_required_verified();
  cout << "n_ext: " << inst_req.size() << endl;

  vector<VkLayerProperties> layer_properties;
  
  if (validation) {
    vector<string> instance_validation_layers =
      {
        "VK_LAYER_GOOGLE_threading",
        "VK_LAYER_LUNARG_parameter_validation",
        "VK_LAYER_LUNARG_object_tracker",
        "VK_LAYER_LUNARG_image",
        "VK_LAYER_LUNARG_core_validation",
        "VK_LAYER_LUNARG_swapchain"
      };
      
    uint32_t n_layer_properties(0);
    check(vkEnumerateInstanceLayerProperties( &n_layer_properties, nullptr ), "vkEnumerateInstanceLayerProperties");
    layer_properties.resize(n_layer_properties);
    check(vkEnumerateInstanceLayerProperties( &n_layer_properties, &layer_properties[0] ), "vkEnumerateInstanceLayerProperties");

    for (int n(0); n < n_layer_properties; ++n)
      for (auto v : instance_validation_layers)
        if (string(layer_properties[n].layerName).find(v) != string::npos)
          enabled_layer_names.push_back(layer_properties[n].layerName);
    inst_req.push_back( VK_EXT_DEBUG_REPORT_EXTENSION_NAME );
  }

  char **inst_req_charp = new char*[inst_req.size()];
  for (int i(0); i < inst_req.size(); ++i)
    inst_req_charp[i] = (char*)inst_req[i].c_str();

        
  VkInstanceCreateInfo ici = {};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pNext = NULL;
  ici.pApplicationInfo = &app_info;
  ici.enabledExtensionCount = inst_req.size();
  ici.ppEnabledExtensionNames = inst_req_charp;
  ici.enabledLayerCount = enabled_layer_names.size();
  ici.ppEnabledLayerNames = &enabled_layer_names[0]; //might need validation layers later

  cout << "creating instance" << endl;
  check( vkCreateInstance( &ici, nullptr, &inst), "Create Instance");
  delete[] inst_req_charp;
}

void VulkanSystem::init_device() {
  auto &vr = Global::vr();
	
  uint32_t n_dev(0);
  check( vkEnumeratePhysicalDevices( inst, &n_dev, NULL ), "vkEnumeratePhysicalDevices");
  vector<VkPhysicalDevice> devices(n_dev);
  check( vkEnumeratePhysicalDevices( inst, &n_dev, &devices[0] ), "vkEnumeratePhysicalDevices");

  if (!Global::inst().HEADLESS)
    phys_dev = (VkPhysicalDevice) vr.get_output_device(inst);
  else
    phys_dev = devices[0]; //select first, could be altered

  vkGetPhysicalDeviceProperties( phys_dev, &prop);
  vkGetPhysicalDeviceMemoryProperties( phys_dev, &mem_prop );
  vkGetPhysicalDeviceFeatures( phys_dev, &features );

  cout << "Device: " << prop.deviceName << endl;
    

  uint32_t n_queue(0);
  vkGetPhysicalDeviceQueueFamilyProperties(  phys_dev, &n_queue, 0);
  vector<VkQueueFamilyProperties> queue_family(n_queue);
  vkGetPhysicalDeviceQueueFamilyProperties( phys_dev, &n_queue, &queue_family[0]);
  if (n_queue == 0) {
    cerr << "Failed to get queue properties.\n" << endl;
    throw "";
  }

  graphics_queue = -1;
  for (int i(0); i < queue_family.size(); ++i) {
    if (queue_family[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      graphics_queue = i;
      break;
    }
  }

  if (graphics_queue < 0) {
    cerr << "no graphics queue" << endl;
    throw "";
  }

  auto dev_ext = vr.get_dev_ext_required_verified();
  // Add additional required extensions
  dev_ext.push_back( VK_KHR_SWAPCHAIN_EXTENSION_NAME );
    
  vector<char *> pp_dev_ext(dev_ext.size());
  for (int i(0); i < dev_ext.size(); ++i)
    pp_dev_ext[i] = (char*) dev_ext[i].c_str();

  //create device
  // Create the device
  VkDeviceQueueCreateInfo dqci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  dqci.queueFamilyIndex = graphics_queue;
  dqci.queueCount = 1;
  float fQueuePriority = 1.0f;
  dqci.pQueuePriorities = &fQueuePriority;

  VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &dqci;
  dci.enabledExtensionCount = dev_ext.size();
  dci.ppEnabledExtensionNames = &pp_dev_ext[0];
  dci.pEnabledFeatures = &features;

  check( vkCreateDevice( phys_dev, &dci, nullptr, &dev ), "vkCreateDevice");

  vkGetDeviceQueue( dev, graphics_queue, 0, &queue );
}


void VulkanSystem::init_descriptor_sets() {
  //Create Layout
  VkDescriptorSetLayoutBinding lb[3] = {};
  lb[0].binding = 0;
  lb[0].descriptorCount = 1;
  lb[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  lb[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  lb[1].binding = 1;
  lb[1].descriptorCount = 1;
  lb[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  lb[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  lb[2].binding = 2;
  lb[2].descriptorCount = 1;
  lb[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  lb[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo desc_set_ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  desc_set_ci.bindingCount = 3;
  desc_set_ci.pBindings = &lb[ 0 ];
  check( vkCreateDescriptorSetLayout( dev, &desc_set_ci, nullptr, &desc_set_layout ), "vkCreateDescriptorSetLayout");


  //Create pool
  size_t NUM_DESCRIPTOR_SETS(256);

  VkDescriptorPoolSize pool_sizes[ 3 ];
  pool_sizes[ 0 ].descriptorCount = NUM_DESCRIPTOR_SETS;
  pool_sizes[ 0 ].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_sizes[ 1 ].descriptorCount = NUM_DESCRIPTOR_SETS;
  pool_sizes[ 1 ].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  pool_sizes[ 2 ].descriptorCount = NUM_DESCRIPTOR_SETS;
  pool_sizes[ 2 ].type = VK_DESCRIPTOR_TYPE_SAMPLER;

  VkDescriptorPoolCreateInfo descpool_ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  descpool_ci.flags = 0;
  descpool_ci.maxSets = NUM_DESCRIPTOR_SETS;
  descpool_ci.poolSizeCount = _countof( pool_sizes );
  descpool_ci.pPoolSizes = &pool_sizes[ 0 ];
  vkCreateDescriptorPool( dev, &descpool_ci, nullptr, &desc_pool );
}



void VulkanSystem::init_shaders() {
  auto &vr = Global::vr();
  //Create Shaders, probably most involved part
  //Init desc sets first
  vector<string> shader_names = {
    "scene",
    "axes",
    "rendermodel"
  };
  if (!Global::inst().HEADLESS)
    shader_names.push_back("companion");
  
  vector<string> stages = {
    "vs",
    "ps"
  };

  int i(0);
  for (auto shader_name : shader_names) {
    for (auto stage : stages) {
      string path = "../shaders/" + shader_name + "_" + stage + ".spv";
      string code = read_all(path);
      cout << path << endl;
      cout << "code size: " << code.size() << endl;
      VkShaderModuleCreateInfo shader_ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
      shader_ci.codeSize = code.size();
      shader_ci.pCode = ( const uint32_t *) &code[0];
      check(vkCreateShaderModule( dev, &shader_ci, nullptr, (stage == "vs") ? &shader_modules_vs[i] : &shader_modules_ps[i] ), "vkCreateShaderModule");
    }
    ++i;
    
  }


  //Create pipelines, first layout
  VkPipelineLayoutCreateInfo pipeline_ci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  pipeline_ci.pNext = NULL;
  pipeline_ci.setLayoutCount = 1;
  pipeline_ci.pSetLayouts = &desc_set_layout;
  pipeline_ci.pushConstantRangeCount = 0;
  pipeline_ci.pPushConstantRanges = NULL;
  check( vkCreatePipelineLayout( dev, &pipeline_ci, nullptr, &pipeline_layout ), "vkCreatePipelineLayout");

  // Create pipeline cache
  VkPipelineCacheCreateInfo pipeline_cache_ci = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
  vkCreatePipelineCache( dev, &pipeline_cache_ci, NULL, &pipeline_cache );

  VkRenderPass render_passes[ PSO_COUNT ] =
	{
      vr.left_eye_fb->render_pass,
      vr.left_eye_fb->render_pass,
      vr.left_eye_fb->render_pass,
      swapchain.render_pass
	};

  //define strides for data used in shaders
  size_t strides[ PSO_COUNT ] =
	{
      sizeof( Pos3Tex2 ),			// PSO_SCENE
      sizeof( float ) * 6,				// PSO_AXES
      sizeof( vr::RenderModel_Vertex_t ),	// PSO_RENDERMODEL
      sizeof( Pos2Tex2 )			// PSO_COMPANION
    };

  VkVertexInputAttributeDescription attr_desc[ PSO_COUNT * 3 ]
  {
    // PSO_SCENE
    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,	0 },
    { 1, 0, VK_FORMAT_R32G32_SFLOAT,	offsetof( Pos3Tex2, texpos ) },
    { 0, 0, VK_FORMAT_UNDEFINED,		0 },
      // PSO_AXES
    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,	0 },
    { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,	sizeof( float ) * 3 },
    { 0, 0, VK_FORMAT_UNDEFINED,		0 },
      // PSO_RENDERMODEL
    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,	0 },
    { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,	offsetof( vr::RenderModel_Vertex_t, vNormal ) },
    { 2, 0, VK_FORMAT_R32G32_SFLOAT,	offsetof( vr::RenderModel_Vertex_t, rfTextureCoord ) },
      // PSO_COMPANION
    { 0, 0, VK_FORMAT_R32G32_SFLOAT,	0 },
    { 1, 0, VK_FORMAT_R32G32_SFLOAT,	sizeof( float ) * 2 },
    { 0, 0, VK_FORMAT_UNDEFINED,		0 },
  };

  // Create the PSOs
  for ( uint32_t pso = 0; pso < shader_names.size(); pso++ )
	{
      VkGraphicsPipelineCreateInfo pipeline_ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
		
      // VkPipelineVertexInputStateCreateInfo
      VkVertexInputBindingDescription binding_ci;
      binding_ci.binding = 0;
      binding_ci.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      binding_ci.stride = strides[ pso ];
		
      VkPipelineVertexInputStateCreateInfo vertexi_ci = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
      for ( uint32_t attr = 0; attr < 3; attr++ )
		{
          if ( attr_desc[ pso * 3 + attr ].format != VK_FORMAT_UNDEFINED )
			{
              vertexi_ci.vertexAttributeDescriptionCount++;
			}
		}
      vertexi_ci.pVertexAttributeDescriptions = &attr_desc[ pso * 3 ];
      vertexi_ci.vertexBindingDescriptionCount = 1;
      vertexi_ci.pVertexBindingDescriptions = &binding_ci;

      // VkPipelineDepthStencilStateCreateInfo
      VkPipelineDepthStencilStateCreateInfo depth_ci = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
      depth_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
      depth_ci.depthTestEnable = ( pso != PSO_COMPANION ) ? VK_TRUE : VK_FALSE;
      depth_ci.depthWriteEnable = ( pso != PSO_COMPANION ) ? VK_TRUE : VK_FALSE;
      depth_ci.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      depth_ci.depthBoundsTestEnable = VK_FALSE;
      depth_ci.stencilTestEnable = VK_FALSE;
      depth_ci.minDepthBounds = 0.0f;
      depth_ci.maxDepthBounds = 0.0f;

      // VkPipelineColorBlendStateCreateInfo
      VkPipelineColorBlendStateCreateInfo colorblend_ci = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
      colorblend_ci.logicOpEnable = VK_FALSE;
      colorblend_ci.logicOp = VK_LOGIC_OP_COPY;
      VkPipelineColorBlendAttachmentState color_attach_state = {};
      color_attach_state.blendEnable = VK_FALSE;
      color_attach_state.colorWriteMask = 0xf;
      colorblend_ci.attachmentCount = 1;
      colorblend_ci.pAttachments = &color_attach_state;

      // VkPipelineColorBlendStateCreateInfo
      VkPipelineRasterizationStateCreateInfo raster_ci = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
      raster_ci.polygonMode = VK_POLYGON_MODE_FILL;
      raster_ci.cullMode = VK_CULL_MODE_BACK_BIT;
      raster_ci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster_ci.lineWidth = 1.0f;

      // VkPipelineInputAssemblyStateCreateInfo
      VkPipelineInputAssemblyStateCreateInfo ia_state_ci = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
      //line triangle decided here!
      ia_state_ci.topology = ( pso == PSO_AXES ) ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      ia_state_ci.primitiveRestartEnable = VK_FALSE;

      // VkPipelineMultisampleStateCreateInfo
      VkPipelineMultisampleStateCreateInfo multisample_ci = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
      multisample_ci.rasterizationSamples = ( pso == PSO_COMPANION ) ? VK_SAMPLE_COUNT_1_BIT : ( VkSampleCountFlagBits ) msaa;
      multisample_ci.minSampleShading = 0.0f;
      uint32_t sample_mask = 0xFFFFFFFF;
      multisample_ci.pSampleMask = &sample_mask;

      // VkPipelineViewportStateCreateInfo
      VkPipelineViewportStateCreateInfo viewport_ci = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
      viewport_ci.viewportCount = 1;
      viewport_ci.scissorCount = 1;

      VkPipelineShaderStageCreateInfo shader_stages[ 2 ] = { };
      shader_stages[ 0 ].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shader_stages[ 0 ].stage = VK_SHADER_STAGE_VERTEX_BIT;
      shader_stages[ 0 ].module = shader_modules_vs[ pso ];
      shader_stages[ 0 ].pName = "VSMain";
		
      shader_stages[ 1 ].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shader_stages[ 1 ].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      shader_stages[ 1 ].module = shader_modules_ps[ pso ];
      shader_stages[ 1 ].pName = "PSMain";

      pipeline_ci.layout = pipeline_layout;

      // Set pipeline states
      pipeline_ci.pVertexInputState = &vertexi_ci;
      pipeline_ci.pInputAssemblyState = &ia_state_ci;
      pipeline_ci.pViewportState = &viewport_ci;
      pipeline_ci.pRasterizationState = &raster_ci;
      pipeline_ci.pMultisampleState = &multisample_ci;
      pipeline_ci.pDepthStencilState = &depth_ci;
      pipeline_ci.pColorBlendState = &colorblend_ci;
      pipeline_ci.stageCount = 2;
      pipeline_ci.pStages = &shader_stages[ 0 ];
      pipeline_ci.renderPass = render_passes[ pso ];

      static VkDynamicState dynamic_states[] =
		{
          VK_DYNAMIC_STATE_VIEWPORT,
          VK_DYNAMIC_STATE_SCISSOR,
		};

      static VkPipelineDynamicStateCreateInfo dynamic_state_ci = {};
      dynamic_state_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamic_state_ci.pNext = NULL;
      dynamic_state_ci.dynamicStateCount = _countof( dynamic_states );
      dynamic_state_ci.pDynamicStates = &dynamic_states[ 0 ];
      pipeline_ci.pDynamicState = &dynamic_state_ci;


      // Create the pipeline
      check( vkCreateGraphicsPipelines( dev, pipeline_cache, 1, &pipeline_ci, NULL, &pipelines[ pso ] ), "vkCreateGraphicsPipelines");
	}
}

// =========== Descriptors ==============
Descriptor::Descriptor() {
  init();
}

Descriptor::~Descriptor() {
  //auto &vk = Global::vk();
   
}

void Descriptor::init() {
  auto &vk = Global::vk();
  //idx = vk.desc_sets.size();
  //vk.desc_sets.push_back(Descriptor());

  VkDescriptorSetAllocateInfo desc_inf = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
  desc_inf.descriptorPool = vk.desc_pool;
  desc_inf.descriptorSetCount = 1;
  desc_inf.pSetLayouts = &vk.desc_set_layout;
  vkAllocateDescriptorSets( vk.dev, &desc_inf, &desc );
}

void Descriptor::register_texture(VkImageView &view) {
  auto &vk = Global::vk();

  VkDescriptorImageInfo img_i = {};
  img_i.imageView = view;
  img_i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkWriteDescriptorSet write_desc[ 1 ] = { };
  write_desc[ 0 ].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_desc[ 0 ].dstSet = desc;
  write_desc[ 0 ].dstBinding = 1;
  write_desc[ 0 ].descriptorCount = 1;
  write_desc[ 0 ].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  write_desc[ 0 ].pImageInfo = &img_i;

  
  vkUpdateDescriptorSets( vk.dev, _countof( write_desc ), write_desc, 0, nullptr );
}

void Descriptor::register_model_texture(VkBuffer &buf, VkImageView &view, VkSampler &sampler) {
  auto &vk = Global::vk();

  VkDescriptorBufferInfo buf_inf = {};
  buf_inf.buffer = buf;
  buf_inf.offset = 0;
  buf_inf.range = VK_WHOLE_SIZE;

  VkDescriptorImageInfo img_inf = {};
  img_inf.imageView = view;
  img_inf.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkDescriptorImageInfo sample_info = {};
  sample_info.sampler = sampler;

  VkWriteDescriptorSet write_desc_set[ 3 ] = { };
  write_desc_set[ 0 ].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_desc_set[ 0 ].dstSet = desc;
  write_desc_set[ 0 ].dstBinding = 0;
  write_desc_set[ 0 ].descriptorCount = 1;
  write_desc_set[ 0 ].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write_desc_set[ 0 ].pBufferInfo = &buf_inf;
  write_desc_set[ 1 ].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_desc_set[ 1 ].dstSet = desc;
  write_desc_set[ 1 ].dstBinding = 1;
  write_desc_set[ 1 ].descriptorCount = 1;
  write_desc_set[ 1 ].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  write_desc_set[ 1 ].pImageInfo = &img_inf;
  write_desc_set[ 2 ].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_desc_set[ 2 ].dstSet = desc;
  write_desc_set[ 2 ].dstBinding = 2;
  write_desc_set[ 2 ].descriptorCount = 1;
  write_desc_set[ 2 ].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write_desc_set[ 2 ].pImageInfo = &sample_info;

  vkUpdateDescriptorSets( vk.dev, _countof( write_desc_set ), write_desc_set, 0, nullptr );
}

void Descriptor::bind() {	
  auto &vk = Global::vk();
  vkCmdBindDescriptorSets( vk.cmd_buffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline_layout, 0, 1, &desc, 0, nullptr );
}

void Swapchain::destroy() { //Needs to be called explicitly
  static int bla(0);
  cout << "swapchain destructor called n times: " << bla++ << endl;
  cout << swapchain << endl;
  if (!n_swap)
    return;
  auto &vk = Global::vk();
  
  for (auto img_ptr : images)
    vkDestroyImageView(vk.dev, img_ptr->view, nullptr);
  for (auto fb : framebuffers)
    vkDestroyFramebuffer(vk.dev, fb, nullptr);
  for (auto sm : semaphores)
    vkDestroySemaphore(vk.dev, sm, nullptr);

  vkDestroyRenderPass(vk.dev, render_pass, nullptr);
  vkDestroySwapchainKHR(vk.dev, swapchain, nullptr);
  vkDestroySurfaceKHR(vk.inst, surface, nullptr);
  n_swap = 0;
}

Swapchain::~Swapchain() {
}

Image &Swapchain::current_img() {
  return *images[current_swapchain_image];
}

void Swapchain::acquire_image() {
  auto &vk = Global::vk();
  //cout << "acquiring image, frame: " << frame_idx << endl;
  try {
    check( vkAcquireNextImageKHR( vk.dev, swapchain, UINT64_MAX, semaphores[ frame_idx ], VK_NULL_HANDLE, &current_swapchain_image ), "vkAcquireNextImageKHR");} catch(...){}
    return;

    
}

void VulkanSystem::init_cmd_pool() {
  // Create the command pool
  VkCommandPoolCreateInfo cmdpoolci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  cmdpoolci.queueFamilyIndex = graphics_queue;
  cmdpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  check( vkCreateCommandPool( dev, &cmdpoolci, nullptr, &cmd_pool ), "vkCreateCommandPool");
}

void VulkanSystem::init_debug_callback() {
  VkDebugReportCallbackCreateInfoEXT ci = {};
  ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
  ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
  ci.pfnCallback = &debug_callback;

  if (create_debug_report_callback_EXT(inst, &ci, nullptr, &callback) != VK_SUCCESS) {
    throw std::runtime_error("failed to set up debug callback!");
  }
}

void VulkanSystem::destroy_debug_callback() {
  auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(inst, "vkDestroyDebugReportCallbackEXT");
  if (func != nullptr)
    func(inst, callback, nullptr);
}

VkCommandBuffer VulkanSystem::cmd_buffer() {
  if (cur_cmd_buffer)
    return cur_cmd_buffer->cmd_buffer;
  for (auto &buf : cmd_buffers) {
    if (buf->ready()) {
      buf->reset();
      cur_cmd_buffer = buf;
      cur_cmd_buffer->begin();
      return cur_cmd_buffer->cmd_buffer;
    }
  }
  //else make a new one
  cur_cmd_buffer = new FencedCommandBuffer();
  cur_cmd_buffer->begin();
  return cur_cmd_buffer->cmd_buffer;
}

int get_mem_type( uint32_t mem_bits, VkMemoryPropertyFlags mem_prop )
{
  auto &vk = Global::vk();
  for ( uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; i++ )
    {
      if ( ( mem_bits & 1 ) == 1)
        {
          // Type is available, does it match user properties?
          if ( ( vk.mem_prop.memoryTypes[i].propertyFlags & mem_prop ) == mem_prop )
            return i;
        }
      mem_bits >>= 1;
    }

  // No memory types matched, return failure
  throw StringException("No Supported Memory Type");
}


void VulkanSystem::end_cmd() {
  if (cur_cmd_buffer == 0)
    throw StringException("cmd still 0 when endcommandbuffer called");
  cur_cmd_buffer->end();
}

void VulkanSystem::submit_swapchain_cmd() {
  // Submit the command buffer
  submit(cur_cmd_buffer->cmd_buffer, cur_cmd_buffer->fence, swapchain.semaphores[ swapchain.frame_idx ]);
  cmd_buffers.push_back(cur_cmd_buffer);
  cur_cmd_buffer = 0;
  swapchain.inc_frame();
}


void VulkanSystem::end_submit_swapchain_cmd() {
  end_cmd();
  submit_swapchain_cmd();
}

void VulkanSystem::submit_cmd() {
  submit(cur_cmd_buffer->cmd_buffer, cur_cmd_buffer->fence);
  cmd_buffers.push_back(cur_cmd_buffer);
  vkWaitForFences(dev, 1, &cur_cmd_buffer->fence, VK_TRUE, 100000000);
  cur_cmd_buffer = 0;
}

void VulkanSystem::end_submit_cmd() {
  end_cmd();
  submit_cmd();
}

