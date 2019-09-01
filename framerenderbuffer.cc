#include "framerenderbuffer.h"
#include "global.h"
#include "util.h"
#include "utilvr.h"

using namespace std;

FrameRenderBuffer::~FrameRenderBuffer() {
  auto &vk = Global::vk();

  vkDestroyRenderPass(vk.dev, render_pass, nullptr);
  vkDestroyFramebuffer(vk.dev, framebuffer, nullptr);
}

void FrameRenderBuffer::start_render_pass() {
	// Start the renderpass
	VkRenderPassBeginInfo renderpassci = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
	renderpassci.renderPass = render_pass;
	renderpassci.framebuffer = framebuffer;
	renderpassci.renderArea.offset.x = 0;
	renderpassci.renderArea.offset.y = 0;
    //cout << "wh:" << width << " " << height << endl;///////////
	renderpassci.renderArea.extent.width = width;
	renderpassci.renderArea.extent.height = height;
	renderpassci.clearValueCount = 2;
	VkClearValue cv[ 2 ];
	cv[ 0 ].color.float32[ 0 ] = 0.0f;
	cv[ 0 ].color.float32[ 1 ] = 0.0f;
	cv[ 0 ].color.float32[ 2 ] = 0.0f;
	cv[ 0 ].color.float32[ 3 ] = 1.0f;
	cv[ 1 ].depthStencil.depth = 1.0f;
	cv[ 1 ].depthStencil.stencil = 0;

	renderpassci.pClearValues = &cv[ 0 ];
	vkCmdBeginRenderPass( Global::vk().cmd_buffer(), &renderpassci, VK_SUBPASS_CONTENTS_INLINE );
}

void FrameRenderBuffer::end_render_pass() {
  vkCmdEndRenderPass( Global::vk().cmd_buffer() );
}

void FrameRenderBuffer::init(int width_, int height_) {
  cout << "init frame render buf: " << width_ << " " << height_ << endl;
	width = width_;
	height = height_;
	auto &vk = Global::vk();
    
	img.init(width, height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1, msaa, false);
	depth_stencil.init(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, msaa, false);

    img_resolve.init_for_copy(width, height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    img_blit.init_for_copy(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    img_data.init(width * height * sizeof(float) * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_COHERENT);
    
    // Create a renderpass
	uint32_t n_attach = 2;
	VkAttachmentDescription att_desc[ 2 ];
	VkAttachmentReference att_ref[ 2 ];
	att_ref[ 0 ].attachment = 0;
	att_ref[ 0 ].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	att_ref[ 1 ].attachment = 1;
	att_ref[ 1 ].layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	att_desc[ 0 ].format = VK_FORMAT_R8G8B8A8_SRGB;
	att_desc[ 0 ].samples = (VkSampleCountFlagBits) msaa;
	att_desc[ 0 ].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	att_desc[ 0 ].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	att_desc[ 0 ].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	att_desc[ 0 ].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	att_desc[ 0 ].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	att_desc[ 0 ].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	att_desc[ 0 ].flags = 0;

	att_desc[ 1 ].format = VK_FORMAT_D32_SFLOAT;
    att_desc[ 1 ].samples = (VkSampleCountFlagBits) msaa;
	att_desc[ 1 ].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	att_desc[ 1 ].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	att_desc[ 1 ].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	att_desc[ 1 ].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	att_desc[ 1 ].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	att_desc[ 1 ].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	att_desc[ 1 ].flags = 0;

	VkSubpassDescription subpassci = { };
	subpassci.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassci.flags = 0;
	subpassci.inputAttachmentCount = 0;
	subpassci.pInputAttachments = NULL;
	subpassci.colorAttachmentCount = 1;
	subpassci.pColorAttachments = &att_ref[ 0 ];
	subpassci.pResolveAttachments = NULL;
	subpassci.pDepthStencilAttachment = &att_ref[ 1 ];
	subpassci.preserveAttachmentCount = 0;
	subpassci.pPreserveAttachments = NULL;

	VkRenderPassCreateInfo renderpass_ci = { };
	renderpass_ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderpass_ci.flags = 0;
	renderpass_ci.attachmentCount = 2;
	renderpass_ci.pAttachments = &att_desc[ 0 ];
	renderpass_ci.subpassCount = 1;
	renderpass_ci.pSubpasses = &subpassci;
	renderpass_ci.dependencyCount = 0;
	renderpass_ci.pDependencies = NULL;

	check( vkCreateRenderPass( vk.dev, &renderpass_ci, NULL, &render_pass ), "vkCreateRenderPass");

	// Create the framebuffer
	VkImageView attachments[ 2 ] = { img.view, depth_stencil.view };
	VkFramebufferCreateInfo fb_ci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
	fb_ci.renderPass = render_pass;
	fb_ci.attachmentCount = 2;
	fb_ci.pAttachments = &attachments[ 0 ];
	fb_ci.width = width;
	fb_ci.height = height;
	fb_ci.layers = 1;
	check( vkCreateFramebuffer( vk.dev, &fb_ci, NULL, &framebuffer), "vkCreateFramebuffer");

	//img.layout = VK_IMAGE_LAYOUT_UNDEFINED;
	//depth_stencil.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    desc.register_texture(img.view);
}

std::vector<float> &FrameRenderBuffer::copy_to_buffer() {
  img.resolve_to_image(img_resolve);
  img_resolve.blit_to_image(img_blit);
  img_blit.copy_to_buffer(img_data);
  float *buf_ptr(0);
  img_data.map(&buf_ptr);
  img_vec.resize(width * height * 4);
  copy(buf_ptr, buf_ptr + img_vec.size(), &img_vec[0]);
  img_data.unmap();
  Global::vk().wait_queue();
  
  return img_vec;
}
