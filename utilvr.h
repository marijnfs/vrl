#ifndef __UTIL_REFRACTOR_H__
#define __UTIL_REFRACTOR_H__

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include "lvulkan.h"
#include <openvr.h>
#include <iostream>
#include <sstream>
#include <SDL.h>
#include <chrono>
#include <thread>
#include <sys/stat.h>

#include "shared/Matrices.h"

#if defined(POSIX)
#include "unistd.h"
#endif

#ifndef _countof
#define _countof(x) (sizeof(x)/sizeof((x)[0]))
#endif

/*
struct StringException : public std::exception {
	StringException(std::string msg_): msg(msg_){}
  template <typename T>
  	StringException(std::string msg_, T t)  {
    std::ostringstream oss;
    oss << msg_ << " " << t << std::endl;
    msg = oss.str();
  }
	char const* what() const throw() {return msg.c_str();}
	~StringException() throw() {}
	std::string msg;
};


struct Timer {
  std::chrono::high_resolution_clock::time_point timepoint;
  double interval;

Timer(float interval_) : interval(interval_) {
    start();
  }

  void start() { timepoint = std::chrono::high_resolution_clock::now(); }

  void wait() {
    while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - timepoint) < std::chrono::duration<double>(interval))
      std::this_thread::sleep_for(std::chrono::duration<int, std::micro>(1));
    start();
  }

  double elapsed() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - timepoint).count();
  }
};


template <typename T>
inline std::ostream &operator<<(std::ostream &out, std::vector<T> in) {
  out << "[";
  typename std::vector<T>::const_iterator it = in.begin(), end = in.end();
  for (; it != end; ++it)
    if (it == in.begin())
      out << *it;
    else
      out << " " << *it;
  return out << "]";
}
*/

inline void ThreadSleep( unsigned long nMilliseconds )
{
#if defined(_WIN32)
  ::Sleep( nMilliseconds );
#elif defined(POSIX)
  usleep( nMilliseconds * 1000 );
#endif
}

template <typename T>
inline void check(VkResult res, std::string str, T other) {
  if (res != VK_SUCCESS) {
    std::map<VkResult, std::string> e;
    e[VK_SUCCESS] = "VK_SUCCESS";
    e[VK_NOT_READY]  = "VK_NOT_READY";
    e[VK_TIMEOUT] = "VK_TIMEOUT";
    e[VK_EVENT_SET] = "VK_EVENT_SET";
    e[VK_EVENT_RESET] = "VK_EVENT_RESET";
    e[VK_INCOMPLETE] = "VK_INCOMPLETE";
    
    e[VK_ERROR_OUT_OF_HOST_MEMORY] = "VK_ERROR_OUT_OF_HOST_MEMORY";
    e[VK_ERROR_OUT_OF_DEVICE_MEMORY] = "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    e[VK_ERROR_INITIALIZATION_FAILED] = "VK_ERROR_INITIALIZATION_FAILED";
    e[VK_ERROR_DEVICE_LOST] = "VK_ERROR_DEVICE_LOST";
    e[VK_ERROR_MEMORY_MAP_FAILED] = "VK_ERROR_MEMORY_MAP_FAILED";
    e[VK_ERROR_LAYER_NOT_PRESENT] = "VK_ERROR_LAYER_NOT_PRESENT";
    e[VK_ERROR_EXTENSION_NOT_PRESENT] = "VK_ERROR_EXTENSION_NOT_PRESENT";
    e[VK_ERROR_FEATURE_NOT_PRESENT] = "VK_ERROR_FEATURE_NOT_PRESENT";
    e[VK_ERROR_INCOMPATIBLE_DRIVER] = "VK_ERROR_INCOMPATIBLE_DRIVER";
    e[VK_ERROR_TOO_MANY_OBJECTS] = "VK_ERROR_TOO_MANY_OBJECTS";
    e[VK_ERROR_FORMAT_NOT_SUPPORTED] = "VK_ERROR_FORMAT_NOT_SUPPORTED";
    e[VK_ERROR_FRAGMENTED_POOL] = "VK_ERROR_FRAGMENTED_POOL";
    std::cerr << str << other << " error: " << e[res] << std::endl;
    throw StringException(e[res]);
  }
}

/* 
template <typename T>
inline T &last(std::vector<T> &v) {
	assert(v.size());
	return v[v.size() - 1];
}
*/

inline void check(VkResult res, std::string str) {
	check(res, str, "");

}

inline void check(vr::EVRInitError err) {
  if ( err != vr::VRInitError_None ) {
    std::cerr << "Unable to init vr: " << vr::VR_GetVRInitErrorAsEnglishDescription(err) << std::endl;
    throw "";
  }
}

inline void sdl_check(int err) {
  if (err < 0) {
    std::cerr << "SDL error: " << SDL_GetError() << std::endl;
    throw "";
  }
}



inline std::string read_all(std::string path) {
	std::ifstream t(path.c_str());
	if (!t)
      throw StringException("failed to open", path);

	std::string str;

	t.seekg(0, std::ios::end);   
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)),
	            std::istreambuf_iterator<char>());
	return str;
}

inline bool exists(const std::string& filename)
{
  struct stat buf;
  if (stat(filename.c_str(), &buf) != -1)
    {
      return true;
    }
  return false;
}

inline void gen_mipmap_rgba( const uint8_t *src, uint8_t *dst, int width, int height, int *width_out, int *height_out )
{
	*width_out = width / 2;
	if ( *width_out <= 0 )
	{
		*width_out = 1;
	}
	*height_out = height / 2;
	if ( *height_out <= 0 )
	{
		*height_out = 1;
	}

	for ( int y = 0; y < *height_out; y++ )
	{
		for ( int x = 0; x < *width_out; x++ )
		{
			int nSrcIndex[4];
			float r = 0.0f;
			float g = 0.0f;
			float b = 0.0f;
			float a = 0.0f;

			nSrcIndex[0] = ( ( ( y * 2 ) * width ) + ( x * 2 ) ) * 4;
			nSrcIndex[1] = ( ( ( y * 2 ) * width ) + ( x * 2 + 1 ) ) * 4;
			nSrcIndex[2] = ( ( ( ( y * 2 ) + 1 ) * width ) + ( x * 2 ) ) * 4;
			nSrcIndex[3] = ( ( ( ( y * 2 ) + 1 ) * width ) + ( x * 2 + 1 ) ) * 4;

			// Sum all pixels
			for ( int nSample = 0; nSample < 4; nSample++ )
			{
				r += src[ nSrcIndex[ nSample ] ];
				g += src[ nSrcIndex[ nSample ] + 1 ];
				b += src[ nSrcIndex[ nSample ] + 2 ];
				a += src[ nSrcIndex[ nSample ] + 3 ];
			}

			// Average results
			r /= 4.0;
			g /= 4.0;
			b /= 4.0;
			a /= 4.0;

			// Store resulting pixels
			dst[ ( y * ( *width_out ) + x ) * 4 ] = ( uint8_t ) ( r );
			dst[ ( y * ( *width_out ) + x ) * 4 + 1] = ( uint8_t ) ( g );
			dst[ ( y * ( *width_out ) + x ) * 4 + 2] = ( uint8_t ) ( b );
			dst[ ( y * ( *width_out ) + x ) * 4 + 3] = ( uint8_t ) ( a );
		}
	}
}

inline float dist(std::vector<float> &v1, std::vector<float> &v2) {
  float d(0);
  std::vector<float>::const_iterator it1(v1.begin()), end1(v1.end()), it2(v2.begin());
  for (; it1 != end1; ++it1, ++it2)
    d += (*it1 - *it2) * (*it1 - *it2);
  return sqrt(d);
}

inline Matrix4 vrmat_to_mat4( const vr::HmdMatrix34_t &matPose )
{
  Matrix4 matrixObj(
    matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
    matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
    matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
    matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
    );
  return matrixObj;
}



struct Pos3Tex2
{
  Vector3 pos;
  Vector2 texpos;
};

struct Pos2Tex2
{
  Vector2 pos;
  Vector2 texpos;
};


inline void print_split(std::vector<float> v, int dim) {
  int d(0);
  for (int i(0); i < v.size(); ++i, ++d) {
    if (d == dim) {
      std::cout << std::endl;
      d = 0;
    }
    std::cout << v[i] << " " ;
  }
}

inline void print_separate(std::vector<float> v, int dim) {
  for (int i(0); i < v.size(); i += v.size() / dim) {
    std::cout << v[i] << " " ;
  }
  std::cout << std::endl;
}

inline void print_nonzero(std::vector<float> v, int dim) {
  for (int i(0); i < v.size(); ++i)
    if (v[i] != 0)
      std::cout << v[i] << " " ;
  std::cout << std::endl;
}


#endif
