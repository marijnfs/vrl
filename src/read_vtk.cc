#include "read_vtk.h"

#include <string>
#include <iostream>


#include <vtkMetaImageReader.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>
using namespace std;


Volume read_vtk(string filename, bool is_segment) {
  vtkSmartPointer<vtkMetaImageReader> reader =
    vtkSmartPointer<vtkMetaImageReader>::New();
  reader->SetFileName(filename.c_str());
  reader->ReleaseDataFlagOn(); 
  reader->Update();
  cout << "width: " << reader->GetWidth() << endl;
  cout << "height: " << reader->GetHeight() << endl;
  
  //    cout << "height: " << reader->GetDepth() << endl;

  cout << "comp: " << reader->GetNumberOfComponents() << endl;
  cout << "pix repr: " << reader->GetPixelRepresentation() << endl;
  vtkImageData *data = reader->GetOutput();
  vector<int> dims(3);
  copy(data->GetDimensions(), data->GetDimensions() + dims.size(), &dims[0]);
  cout << "scalar type: " << data->GetScalarType() << endl;
  cout << "dims: " << dims << endl;
  
  int n_channels = reader->GetNumberOfComponents();
  int width(dims[0]), height(dims[1]), depth(dims[2]);

  //Copy data into volume
  Volume vol(VolumeShape{depth, n_channels, width, height});
  vector<float> vec_data(vol.size());
  if (is_segment) {
    cout << "reading segmentation" << endl;
    int n_classes = 5; //for BRATS
    vol.reshape(VolumeShape{depth, n_classes, width, height});
    vec_data.resize(vol.size());
    if (data->GetScalarType() == 3)
      copy_vtk_to_class_vector<unsigned char>(data, vec_data, depth, width, height, n_classes);
    else if (data->GetScalarType() == 4)
      copy_vtk_to_class_vector<short>(data, vec_data, depth, width, height, n_classes);
  } else if (data->GetScalarType() == 4) {
    cout << "short, assuming data" << endl;
    copy_vtk_to_vector<short>(data, vec_data, depth, width, height, n_channels);
    
    //Post Process
    vector<bool> mask(vec_data.size());
    for (int n(0); n < mask.size(); ++n) if (vec_data[n] != 0) mask[n] = true;
    normalize_masked(&vec_data, mask);
  }
  

  
  vol.from_vector(vec_data);
  return vol;
}
