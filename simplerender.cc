#include "simplerender.h"

using namespace std;

SimpleRender::SimpleRender(int width_, int height_) :
  width(width_),
  height(height_)
{
  left_fb.init(width, height);
  right_fb.init(width, height);
}

void SimpleRender::render(Scene &scene) {
  
}
