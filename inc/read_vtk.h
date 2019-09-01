#ifndef __READ_VTK_H__
#define __READ_VTK_H__

#include <string>
#include "volume.h"
#include <vtkImageData.h>
#include <iostream>

template <typename T>
inline void copy_vtk_to_vector(vtkImageData *data, std::vector<float> &vec, int depth, int width, int height, int n_channels) {
  std::vector<float>::iterator it = vec.begin();
  
  for (size_t z(0); z < depth; ++z)
    for (size_t c(0); c < n_channels; ++c)
      for (size_t y(0); y < height; ++y) {
	for (size_t x(0); x < width; ++x) {
	  //std::cout << (float)*(reinterpret_cast<T*>(data->GetScalarPointer(x, y, z)) + c) << " ";
	  *it = (float)*(reinterpret_cast<T*>(data->GetScalarPointer(x, y, z)) + c) ;
	  ++it;
	}
	//std::cout << std::endl;
      }
}

template <typename T>
inline void copy_vtk_to_class_vector(vtkImageData *data, std::vector<float> &vec, int depth, int width, int height, int n_channels) {
  std::vector<float>::iterator it = vec.begin();
  
  for (size_t z(0); z < depth; ++z)
    for (size_t y(0); y < height; ++y) {
      for (size_t x(0); x < width; ++x) {
	//set value in right channel to 1
	int c = *reinterpret_cast<T*>(data->GetScalarPointer(x, y, z));
	vec[z * n_channels * width * height + c * width * height + y * width + x] = 1;
	
      }
    }
  
}

Volume read_vtk(std::string filename, bool is_segment = false);

#endif
