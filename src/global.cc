#include "global.h"

Global *Global::s_global = 0;

Global::Global() : valid(false)
{
  init_global();
}

void Global::init_global() {
}


void Global::s_init() {
  s_global = new Global();
  s_global->init_global();
}

Global &Global::inst() {
  if (!Global::s_global)
    Global::s_global = new Global();
  return *Global::s_global;
}

bool &Global::validation() {
  return Global::inst().valid;
}
