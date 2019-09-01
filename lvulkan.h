#if defined( _WIN32 )
   #define VK_USE_PLATFORM_WIN32_KHR
#else
   #define SDL_VIDEO_DRIVER_X11
   #define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <vulkan/vulkan.h>
