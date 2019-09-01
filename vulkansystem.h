#ifndef __VULKANSYSTEM_H__
#define __VULKANSYSTEM_H__

#include "lvulkan.h"
#include <vector>
#include <queue>
#include "buffer.h"
#include "shared/Matrices.h"
#include "flywheel.h"
#include "util.h"

inline Matrix4 glm_to_mat4(glm::mat4 m) {
  //auto m = &mat[0];
  return Matrix4(m[0][0], m[0][1], m[0][2], m[0][3],
		m[1][0], m[1][1], m[1][2], m[1][3], 
		m[2][0], m[2][1], m[2][2], m[2][3], 
          m[3][0], m[3][1], m[3][2], m[3][3]);
}

enum PSO //shader files
{
  PSO_SCENE = 0,
  PSO_AXES,
  PSO_RENDERMODEL,
  PSO_COMPANION,
  PSO_COUNT
};

struct FencedCommandBuffer {
  VkCommandBuffer cmd_buffer;
  VkFence fence;

  FencedCommandBuffer();
  void begin();
  void end();
  bool ready();
  void reset();
  void init();

};


/*
struct RenderModel {
	Buffer mat_buffer;
	Buffer vertex_buf, index_buf;

	VkDescriptorSet desc_set;

	void init();
	};*/

struct Descriptor {
	int idx = 0;
	VkDescriptorSet desc;
  VkImageView *view_ptr = 0;
  VkSampler *sampler_ptr = 0;
  VkBuffer *buffer_ptr = 0;


  Descriptor();
  ~Descriptor();
  void init();
  void register_texture(VkImageView &view);
  void register_model_texture(VkBuffer &buf, VkImageView &view, VkSampler &sampler);

  void bind();
};



struct Swapchain {
  ~Swapchain();
  
  VkSurfaceKHR surface = 0;
  VkSwapchainKHR swapchain = 0;

  VkRenderPass render_pass = 0;
  
  std::vector< VkImage > vk_images;
  std::vector< Image* > images;
  //std::vector< VkImageView > views;
  std::vector< VkFramebuffer > framebuffers;
  std::vector< VkSemaphore > semaphores;

  uint32_t n_swap = 0;
  uint32_t frame_idx = 0;
  uint32_t current_swapchain_image = 0;
  
  //VkRenderPass renderpass, companion_renderpass;


  void to_present();
  void init();
  void destroy();
  //void to_present(int i);
  //void to_colour_optimal(int i);
  //void to_present_optimal(int i);

  void acquire_image();
  Image &current_img();
  void begin_render_pass(uint32_t width, uint32_t height);
  void end_render_pass();

  void inc_frame();
};

struct VulkanSystem {
  VkInstance inst;

  VkDevice dev;
  VkPhysicalDevice phys_dev;
  VkPhysicalDeviceProperties prop;
  VkPhysicalDeviceMemoryProperties mem_prop;
  VkPhysicalDeviceFeatures features;

  VkQueue queue;

  int graphics_queue;
  //uint32_t msaa = 1;

  VkCommandPool cmd_pool = 0;
  Swapchain swapchain;


  VkDescriptorPool desc_pool = 0;
  VkDescriptorSetLayout desc_set_layout = 0;
  std::vector<VkDescriptorSet> desc_sets;

  bool validation = true;
  VkDebugReportCallbackEXT callback = 0;
  
  //Buffer scene_constant_buffer[2]; //for both eyes

	//VkImage scene_img;
	//VkDeviceMemory scene_img_mem;
	//VkImageView scene_img_view;

	// Buffer scene_staging;
	// VkBuffer scene_staging_buffer;
	// VkDeviceMemory scene_staging_buffer_memory;

	// VkSampler scene_sampler;

  //Shader stuff
  VkShaderModule shader_modules_vs[PSO_COUNT], shader_modules_ps[PSO_COUNT];
  VkPipeline pipelines[PSO_COUNT];
  VkPipelineLayout pipeline_layout = 0;
  VkPipelineCache pipeline_cache = 0;

  std::deque< FencedCommandBuffer* > cmd_buffers;
  FencedCommandBuffer *cur_cmd_buffer = 0;
  
  VkSampler sampler = 0;
  
  VulkanSystem();
  ~VulkanSystem();
  
  void init(); //general init
  void init_instance();
  void init_device();
  void init_cmd_pool();
  void init_descriptor_sets();
  void init_swapchain();
  void init_shaders();
  void init_texture_maps();
  void init_debug_callback();

  void destroy_debug_callback();
    
  void setup();

  void submit(VkCommandBuffer cmd, VkFence fence, VkSemaphore semaphore = 0);
  void wait_queue();
  void wait_idle();
  void flush_cmd();
  
  void add_desc_set();

 //void init_vulkan();

  void end_cmd();
  void submit_cmd();
  void end_submit_cmd();
  void submit_swapchain_cmd();
  void end_submit_swapchain_cmd();
  
  VkCommandBuffer cmd_buffer();
  
  //void img_to_colour_optimal(VkImage &img);
  //void swapchain_to_present(int i);
};

#include "scene.h"
struct GraphicsObject {
  Descriptor desc_left, desc_right;
 
  int n_vertex = 0, n_index = 0;
  Buffer vertex_buf;
  Buffer index_buf;
  
  Matrix4 *mvp_left = 0, *mvp_right = 0;
  Buffer mvp_buffer_left, mvp_buffer_right;
  
  //Image texture;
  
  std::vector<float> vertices;
  std::vector<uint16_t> indices;
  
  GraphicsObject();
  virtual ~GraphicsObject();
  
  virtual void render(Matrix4 &mvp, bool right);
    void init_buffers();
  void add_vertex(float x, float y, float z, float tx, float ty);  
  
};

struct GraphicsCanvas : public GraphicsObject {
  std::string texture; //flywheel is responsible for keeping image resources

  GraphicsCanvas();
  GraphicsCanvas(std::string texture_);

  void init();
  
  //void render(Matrix4 &mvp, bool right);
  
  void change_texture(std::string texture_) {
    if (texture == texture_) return;
    texture = texture_;
    auto *img = ImageFlywheel::image(texture);
    desc_left.register_texture(img->view);
   desc_right.register_texture(img->view);
  }
  
};

struct GraphicsCube : public GraphicsObject {
  std::string texture;
  float width=1, height=1, depth=1;
  bool balanced = true;
  GraphicsCube();

  void init();
  
  //virtual void render(Matrix4 &mvp, bool right);
  void change_texture(std::string texture_) {
    if (texture == texture_) return;
    texture = texture_;
    auto *img = ImageFlywheel::image(texture);
    desc_left.register_texture(img->view);
    desc_right.register_texture(img->view);
  }

  void change_dim(float width_, float height_, float depth_);
  void set_vertices();
};

struct DrawVisitor : public ObjectVisitor {
  std::vector<GraphicsObject*> gobs;
  
  Matrix4 mvp;
  bool right = false;

  ~DrawVisitor() {
    for (auto g : gobs)
      delete g;
  }
  
  template <typename T>
    void check_size_and_type(int i) {
    if (i >= gobs.size()) {
      gobs.resize(i+1);
      gobs[i] = new T();
      return;
    }
    
    if (dynamic_cast<T*>(gobs[i]) == NULL) {
      delete gobs[i];
      gobs[i] = new T();
    }
  }

  template <typename T>
    T& gob(int i) {
    check_size_and_type<T>(i);
    return *dynamic_cast<T*>(gobs[i]);
  }
  
  void visit(Canvas &canvas) {
    auto &gcanvas = gob<GraphicsCanvas>(i);
    gcanvas.change_texture(canvas.tex_name);
    
    auto mat = mvp * glm_to_mat4(canvas.to_mat4());

    //std::cout << "render canvas" << std::endl;
    gcanvas.render(mat, right);
  }

  void visit(Box &box) {
    auto &gbox = gob<GraphicsCube>(i);
    gbox.change_texture(box.tex_name);
    gbox.change_dim(box.width, box.height, box.depth);
    //std::cout << "drawing box" << std::endl;
    auto mat = mvp * glm_to_mat4(box.to_mat4());
    gbox.render(mat, right);
  }
  
  void visit(Controller &controller) {
    auto &gbox = gob<GraphicsCube>(i);
    gbox.change_texture("red-checker.png");
        
    gbox.change_dim(.005, .005, .005);
    auto controller_mat = glm_to_mat4(controller.to_mat4());
    auto mat = mvp * controller_mat;
    
    //std::cout << "controller: " << controller.p << std::endl;
 
    gbox.render(mat, right);
  }
  
  void visit(Point &point) {
  }
  
  void visit(HMD &hmd) {
  }
                              
};

int get_mem_type( uint32_t mem_bits, VkMemoryPropertyFlags mem_prop );


inline VkResult create_debug_report_callback_EXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
  auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pCallback);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

/*
typedef VkBool32 (VKAPI_PTR *PFN_vkDebugReportCallbackEXT)(
                   VkDebugReportFlagsEXT                       flags,
                   VkDebugReportObjectTypeEXT                  objectType,
                   uint64_t                                    object,
                   size_t                                      location,
                   int32_t                                     messageCode,
                   const char*                                 pLayerPrefix,
                   const char*                                 pMessage,
                   void*                                       pUserData);
*/
static VkBool32 VKAPI_PTR debug_callback(
                                                    VkDebugReportFlagsEXT flags,
                                                    VkDebugReportObjectTypeEXT objType,
                                                    uint64_t obj,
                                                    size_t location,
                                                    int32_t code,
                                                    const char* layerPrefix,
                                                    const char* msg,
                                                    void* userData) {

  std::cout << "In Debug Callback: " << std::endl;
  std::cerr << "validation layer: " << msg << std::endl;
  // throw StringException(msg);
  return VK_FALSE;
}

#endif
