#ifndef __SCRIPT_H__
#define __SCRIPT_H__

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

extern "C"{
#include <lua5.3/lua.h>
#include <lua5.3/lauxlib.h>
#include <lua5.3/lualib.h>
}

typedef int (*LuaFunc)(lua_State *L);

struct Script {
  Script();
  ~Script();

  void run(std::string filename);
  void run_buffer(std::vector<uint8_t> &data);
  void run_interactive();
  void init();

  void call_callback();

  void register_c_func(std::string name, LuaFunc f);
  //regular commands

  lua_State *L;
  std::vector<int> funcs;
};

int add_hmd(lua_State *L);

#endif
