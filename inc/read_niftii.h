#ifndef __READ_NIFTII_H__
#define __READ_NIFTII_H__

#include <nifti1_io.h>
//#include <nifti1.h>
#include <vector>
#include <fstream>
#include <string>
#include <cstdint>

#include "volume.h"

struct NiftiVolume {
  nifti_1_header hdr;

  std::vector<float> data;

  NiftiVolume(std::string filename);

  Volume get_volume();
};


#endif
