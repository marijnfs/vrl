#include "script.h"
#include "util.h"
#include "global.h"

using namespace std;


/*
int i, n;
// 1st argument must be a table (t)
luaL_checktype(L,1, LUA_TTABLE);

n = luaL_getn(L, 1);  // get size of table
for (i=1; i<=n; i++)
  {
    lua_rawgeti(L, 1, i);  // push t[i]
    int Top = lua_gettop(L);
    Position[i-1] = lua_tonumber(L, Top);
    if (i>3) { break; }
  }
*/

int add_hmd(lua_State *L) {
  int nargs = lua_gettop(L);
  Global::scene().add_hmd();
  return 0;
}

int add_controller(lua_State *L) {
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  Global::scene().add_object(name, new Controller(true));
  return 0;
}

int add_box(lua_State *L) {
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  Global::scene().add_box(name);

  return 0;
}

int add_variable(lua_State *L) {
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  Global::scene().add_variable(name, new FreeVariable());

  return 0;
}

int add_mark_variable(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  Global::scene().add_variable(name, new MarkVariable());
 
  return 0;
}

int set_variable(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 2) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  float val = lua_tonumber(L, 1);
  Global::scene().variable<Variable>(name).set_value(val);
 
  return 0;
}

int set_pos(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 4) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  float x = lua_tonumber(L, 2);
  float y = lua_tonumber(L, 3);
  float z = lua_tonumber(L, 4);
  Global::scene().set_pos(name, Pos(x, y, z));
 
  return 0;
}

int set_texture(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 2) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  string tex = lua_tostring(L, 2);
  Global::scene().find<Box>(name).set_texture(tex);
  return 0;
}

int set_dim(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 4) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  float x = lua_tonumber(L, 2);
  float y = lua_tonumber(L, 3);
  float z = lua_tonumber(L, 4);
  Global::scene().find<Box>(name).set_dim(x, y, z);
  return 0;
}

int set_reward(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  float r = lua_tonumber(L, 1);
  Global::scene().set_reward(r);
  return 0;
}

int add_click_trigger(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 2) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  string callback = lua_tostring(L, 2);
  Global::scene().add_trigger(new ClickTrigger(Global::scene()(name)), callback);
  
  //todo create callback, anonymous?
  return 0;
}

int add_next_trigger(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  string callback = lua_tostring(L, 1);
  Global::scene().add_trigger(new NextTrigger(), callback);
  
  //todo create callback, anonymous?
  return 0;
}

int add_inbox_trigger(lua_State *L) { 
  auto &scene = Global::scene();
  int nargs = lua_gettop(L);
  if (nargs != 3) throw StringException("not enough arguments");
  string box = lua_tostring(L, 1);
  string target = lua_tostring(L, 2);
  string callback = lua_tostring(L, 3);
  
  scene.add_trigger(new InBoxTrigger(scene(box), scene(target)), callback);

  //todo create callback, anonymous?
  return 0;
}

int clear_objects(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 0) throw StringException("not enough arguments");
  Global::scene().clear_objects();
  return 0;
}

int clear_triggers(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 0) throw StringException("not enough arguments");
  Global::scene().clear_triggers();
  return 0;
}

int clear(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 0) throw StringException("not enough arguments");
  Global::scene().clear();
  return 0;
}

int clear_scene(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 0) throw StringException("not enough arguments");
  Global::scene().clear_scene();
  return 0;
}

int start_recording(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 0) throw StringException("not enough arguments");
  Global::scene().start_recording();
  return 0;
}

int stop(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 0) throw StringException("not enough arguments");
  Global::scene().stop = true;
  return 0;
}

int choose(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 2) throw StringException("not enough arguments");
  int n = lua_tointeger(L, 1);
  int exclude = lua_tointeger(L, 2);

  int new_choice = rand() % n + 1;
  while (new_choice == exclude)
    new_choice = rand() % n + 1;
  lua_pushinteger(L, new_choice); //lua indexing
  return 1;
}

int is_clicked(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 1) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  
  lua_pushboolean(L, Global::scene().find<Controller>(name).clicked == true);
  return 1;
}



int register_function(lua_State *L) { 
  int nargs = lua_gettop(L);
  if (nargs != 2) throw StringException("not enough arguments");
  string name = lua_tostring(L, 1);
  int f = luaL_ref(L, LUA_REGISTRYINDEX);
  Global::scene().register_function(name, [L,f]()->void{
      //cout << "calling anonymous callback " << f << endl;
      lua_rawgeti(L, LUA_REGISTRYINDEX, f);
      if (lua_pcall(L, 0, LUA_MULTRET, 0))
        throw StringException(lua_tostring(L, -1));
      
    });
  //todo create callback, anonymous?
  return 0;
}


Script::Script() {
  init();
}

Script::~Script() {
  lua_close(L);
}

void Script::init() {
  L = luaL_newstate();   /* opens Lua */
  luaL_openlibs(L);             /* opens the basic library */
  
  register_c_func("add_hmd", &add_hmd);
  register_c_func("add_controller", &add_controller);
  register_c_func("add_inbox_trigger", &add_inbox_trigger);
  register_c_func("add_click_trigger", &add_click_trigger);
  register_c_func("add_next_trigger", &add_next_trigger);
  register_c_func("add_variable", &add_variable);
  register_c_func("add_mark_variable", &add_mark_variable);
  register_c_func("add_box", &add_box);
  register_c_func("clear", &clear);
  register_c_func("clear_triggers", &clear_triggers);
  register_c_func("clear_objects", &clear_objects);
  register_c_func("clear_scene", &clear_scene);
  register_c_func("set_dim", &set_dim);
  register_c_func("set_texture", &set_texture);
  register_c_func("set_pos", &set_pos);
  register_c_func("set_variable", &set_variable);

  register_c_func("set_reward", &set_reward);
  register_c_func("start_recording", &start_recording);
  register_c_func("stop", &stop);
  register_c_func("choose", &choose);
  
  register_c_func("register_function", register_function);
  register_c_func("is_clicked", is_clicked);
}



void Script::run(string filename) {
  if (luaL_dofile(L, filename.c_str()))
    throw StringException(lua_tostring(L, -1));
}

void Script::run_buffer(vector<uint8_t> &data) {
  int error = luaL_loadbuffer(L, reinterpret_cast<char*>(&data[0]), data.size(), "buffer");
    lua_pcall(L, 0, 0, 0);

    if (error) {
      fprintf(stderr, "%s", lua_tostring(L, -1));
      lua_pop(L, 1);  /* pop error message from the stack */
      throw StringException("LUA script error");
    }
}

void Script::run_interactive() {
  string s;
  int error;
  
  while (getline(cin, s)) {
    error = luaL_loadbuffer(L, s.c_str(), s.size(), "line");
    lua_pcall(L, 0, 0, 0);

    if (error) {
      fprintf(stderr, "%s", lua_tostring(L, -1));
      lua_pop(L, 1);  /* pop error message from the stack */
    }
  }
}

void Script::call_callback() {
  for (auto f : funcs) {
    lua_rawgeti(L, LUA_REGISTRYINDEX, f);
    lua_pcall(L, 0, LUA_MULTRET, 0);
  }
}

void Script::register_c_func(std::string name, LuaFunc f) {
  lua_register(L, name.c_str(), f);
}
