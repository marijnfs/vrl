#include <string>
#include "H5Cpp.h"

#include <iostream>

void save_h5(std::vector<float> &values, std::string filename, std::vector<int> sizes)
{
  std::vector<hsize_t> dims(sizes.size()), chunk_dims(sizes.size());
  for (int i(0); i < dims.size(); ++i) chunk_dims[i] = dims[i] = sizes[i];
  std::cout << sizes[0] << std::endl;
  chunk_dims[0] = 50;

  // Create a new file using the default property lists.
  H5::H5File file(filename, H5F_ACC_TRUNC);
  
  // Create the data space for the dataset.
  H5::DataSpace *dataspace = new H5::DataSpace(dims.size(), &dims[0]);
  
  // Modify dataset creation property to enable chunking
  H5::DSetCreatPropList  *plist = new H5::DSetCreatPropList;
  plist->setChunk(chunk_dims.size(), &chunk_dims[0]);
  
  // Set ZLIB (DEFLATE) Compression using level 6.
  // To use SZIP compression comment out this line.
  plist->setDeflate(6);
  
  // Uncomment these lines to set SZIP Compression
  // unsigned szip_options_mask = H5_SZIP_NN_OPTION_MASK;
  // unsigned szip_pixels_per_block = 16;
  // plist->setSzip(szip_options_mask, szip_pixels_per_block);
  
  // Create the dataset.
  H5::DataSet *dataset = new H5::DataSet(file.createDataSet( "data",
                                                             H5::PredType::IEEE_F32BE, *dataspace, *plist) );
  // Write data to dataset.
  dataset->write(&values[0], H5::PredType::NATIVE_FLOAT);
  
  // Close objects and file.  Either approach will close the HDF5 item.
  file.close();
  delete dataspace;
  delete dataset;
  delete plist;
  
}
