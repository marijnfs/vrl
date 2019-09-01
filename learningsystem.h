#ifndef __LEARNING_SYSTEM__
#define __LEARNING_SYSTEM__

#include <string>
#include "walk.h"
#include "scene.h"

struct Learner {
  std::string path = "/home/marijnfs/data/";

  
  Learner() {}

  void learn() {
    auto files = walk(path, "*.rec");

    Recording recording;
    Scene scene;

    recording.load(files[0], &scene);

    
  }
  
};

#endif
