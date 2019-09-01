#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <vector>
#include <string>
#include "lvulkan.h"

enum Location {
	HOST,
	DEVICE,
	HOST_COHERENT
};

struct Buffer {
  VkBuffer buffer = 0;
  VkDeviceMemory memory = 0;
  size_t n = 0; //number of bytes
  
  Buffer();
  Buffer(size_t size, VkBufferUsageFlags usage, Location loc);
  ~Buffer();
  void destroy();
  
  template <typename T>
  Buffer(std::vector<T> &init_data, VkBufferUsageFlags usage, Location loc);
  
  template <typename T>
  Buffer(T init_data[], int n_, VkBufferUsageFlags usage, Location loc);

  template <typename T>
  void init(std::vector<T> &init_data, VkBufferUsageFlags usage, Location loc);
  
  template <typename T>
  void init(T init_data[], int n_, VkBufferUsageFlags usage, Location loc);
  
  void init(size_t size, VkBufferUsageFlags usage, Location loc);

  template <typename T>
  void update(std::vector<T> &init_data);
  
  template <typename T>
  void map(T **ptr);
  void unmap();
  
  size_t size() { return n; }
  template <typename T>
  std::vector<T> get_data();

};

struct ViewedBuffer {
    Buffer buffer;
    VkBufferView buffer_view;
};

struct Image {
    VkImage img = 0;
    VkDeviceMemory mem = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkAccessFlags access_flags = 0;
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  
  VkImageView view = 0;
    VkSampler sampler = 0;

    unsigned width = 0, height = 0;
    int mip_levels = 1;

  Buffer staging_buffer;
  
  Image(VkImage img_, VkFormat format_, VkImageAspectFlags aspect_);
  
    Image();
  Image(int width, int height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT, int msaa_sample_count = 1);
  Image(std::string path, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  ~Image();

  void init(int width_, int height_, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect_, int mip_levels_, int msaa_sample_count, bool make_sampler);
  void init_from_img(std::string img_path, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect);
  void init_for_copy(int width, int height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags mem_flags);

  void barrier(VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage, VkImageLayout new_layout);
  
  void blit_to_image(Image &dst_image);
  void resolve_to_image(Image &dst);
  
  void copy_to_buffer(Buffer &buf);

  template <typename T>
  std::vector<T> get_data() {
    return staging_buffer.get_data<T>();
  }

  VkSubresourceLayout subresource_layout();
  
};



#endif
