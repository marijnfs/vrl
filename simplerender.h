#ifndef __SIMPLE_RENDER_H__
#define __SIMPLE_RENDER_H__

#include "vrsystem.h"

//A simple renderer for both eyes, without the openvr (for learning)

struct SimpleRender {
  SimpleRender(int width, int height);

  void render(Scene &scene);

int width = 0, height = 0;
  FrameRenderBuffer left_fb, right_fb;
};

#endif
