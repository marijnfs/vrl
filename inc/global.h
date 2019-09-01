#ifndef __GLOBAL_H__
#define __GLOBAL_H__

struct Global {
  Global();
  void init_global();

  static bool &validation();
  static void s_init();
  static Global &inst();


  static Global *s_global;

  bool valid;
};

#endif
