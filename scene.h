#ifndef __SCENE_H__
#define __SCENE_H__

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <sstream>
#include <fstream>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

typedef glm::fquat fquat;
//typedef typename std::vector<float> Pos;
typedef typename glm::fvec3 Pos;

#include "bytes.h"
#include "snap.capnp.h"
#include "util.h"

struct Variable;
struct Trigger;

typedef typename std::function<void(void)> void_function;

inline std::string concat(std::string name, int id) {
  std::ostringstream oss;
  oss << name << id;
  return oss.str();
}

inline void write_all_bytes(std::string filename, std::vector<char> &bytes) {
  std::ofstream ofs(filename.c_str(), std::ios::binary | std::ios::out);

  ofs.write(&bytes[0], bytes.size());
}

inline std::vector<char> read_all_bytes(std::string filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::ate);
  std::ifstream::pos_type pos = ifs.tellg();
  std::vector<char> data(pos);
  
  ifs.seekg(0, std::ios::beg);
  ifs.read(&data[0], pos);

  return data;
}

struct Canvas;
struct Controller;
struct Point;
struct HMD;
struct Box;
struct Object;
struct Scene;

struct ObjectVisitor {
  int i = 0;

  virtual void visit(Canvas &canvas) {};
  virtual void visit(Controller &controller) {};
  virtual void visit(Point &point) {};
  virtual void visit(HMD &hmd) {};
  virtual void visit(Box &box) {};
};

struct JSONVisitor : public ObjectVisitor {
  Scene &scene;
  std::ostream &out;
  bool first = true;

JSONVisitor(Scene &scene_, std::ostream &out_) : scene(scene_), out(out_) {}
  
  virtual void visit(Canvas &canvas);
  virtual void visit(Controller &controller);
  virtual void visit(Point &point);
  virtual void visit(HMD &hmd);
  virtual void visit(Box &box);

  bool object_json_output(Object &o);
};


struct Object {
  bool changed = true;
  int nameid = -1;
  
  Pos p;
  fquat quat;
  
  Object() : p(3) {}
  virtual ~Object() {}
  
  virtual Object *copy() {return new Object(*this); }
  
  void set_pos(Pos pos) { p = pos; changed = true; }
  void from_axis(float x, float y, float z, float angle) { //angle in degrees
    quat = glm::angleAxis<float>(angle, glm::vec3(x, y, z));
    changed = true;
  }

  void look_at(Pos to, Pos up) {
    quat = glm::quat_cast(glm::lookAt(p, to, up));
  }
  
  float angle() {
    return glm::angle(quat);
  }

  glm::mat4 to_mat4() {
    auto m = glm::toMat4(quat);
    m[3][0] = p[0];
    m[3][1] = p[1];
    m[3][2] = p[2];
    m[3][3] = 1.0;
    return m;
  }

  void from_mat4(glm::fmat4 m) {
    auto ptr = (float*)glm::value_ptr(m);
    quat = glm::quat_cast(m);
    p[0] = ptr[12];
    p[1] = ptr[13];
    p[2] = ptr[14];
  }

  virtual void update() {
  }

  virtual void visit(ObjectVisitor &visitor) {
  }

  
  
  virtual void serialise(cap::Object::Builder builder) {
    auto q = builder.initQuat();
    q.setA(quat[0]);
    q.setB(quat[1]);
    q.setC(quat[2]);
    q.setD(quat[3]);

    auto p_ = builder.initPos();
    p_.setX(p[0]);
    p_.setY(p[1]);
    p_.setZ(p[2]);

    builder.setNameId(nameid);
  }

  virtual void deserialise(cap::Object::Reader reader) {
    nameid = reader.getNameId();

    auto q = reader.getQuat();
    quat[0] = q.getA();
    quat[1] = q.getB();
    quat[2] = q.getC();
    quat[3] = q.getD();

    auto p_ = reader.getPos();
    p[0] = p_.getX();
    p[1] = p_.getY();
    p[2] = p_.getZ();    
  }
};

Object *read_object(cap::Object::Reader reader);

struct Box : public Object {
  float width = 1, height = 1, depth = 1;
  std::string tex_name = "stub.png";

  ~Box() {}
  
  void serialise(cap::Object::Builder builder) {
    Object::serialise(builder);
    auto b = builder.initBox();
    b.setW(width);
    b.setH(height);
    b.setD(depth);
    b.setTexture(tex_name);
  }

  void deserialise(cap::Object::Reader reader) {
    Object::deserialise(reader);
    auto b = reader.getBox();

    width = b.getW();
    height = b.getH();
    depth = b.getD();
    tex_name = b.getTexture();
  }
  
  void set_dim(float w, float h, float d) {
    width = w;
    height = h;
    depth = d;
    changed = true;
  }

  void set_texture(std::string name) {
    changed = true;
    tex_name = name;
  }

  Object *copy() {
    return new Box(*this);
  }
  
  void visit(ObjectVisitor &visitor) {
    visitor.visit(*this);
  }

};

struct Point : public Object {
  ~Point() {}
  
  void serialise(cap::Object::Builder builder) {
    Object::serialise(builder);
    builder.setPoint();
  }

  Object *copy() {
    return new Point(*this);
  }
  
  void visit(ObjectVisitor &visitor) {
    visitor.visit(*this);
  }

};

struct Canvas : public Object {
  std::string tex_name = "stub.png";

 Canvas() {}
 Canvas(std::string tex_name_) : tex_name(tex_name_) {}
 ~Canvas() {}
 
  Object *copy() {
    return new Canvas(*this);
  }

  void serialise(cap::Object::Builder builder) {
    Object::serialise(builder);
    builder.setCanvas(tex_name);
  }

  void deserialise(cap::Object::Reader reader) {
    Object::deserialise(reader);
    tex_name = reader.getCanvas();
  }

  void set_texture(std::string name) {
    changed = true;
    tex_name = name;
  }

  void visit(ObjectVisitor &visitor) {
    visitor.visit(*this);
  }

};

struct HMD : public Object {
  bool tracked = true;

  ~HMD() {}
  
  void update();

  void serialise(cap::Object::Builder builder) {
    Object::serialise(builder);
    builder.setHmd();
  }

  void deserialise(cap::Object::Reader reader);
  
  Object *copy() {
    return new HMD(*this);
  }

  void visit(ObjectVisitor &visitor) {
    visitor.visit(*this);
  }

};

struct Controller : public Object {
  bool right = true;
  bool pressed = false; //true if pressed
  bool clicked = false; //true if pressed and previously unpressed (so only one frame)
  bool tracked = true;

  Controller(){}
 Controller(bool right_) : right(right_) {}
  ~Controller() {}

  Object *copy() {
    return new Controller(*this);
  }

  void update();

  
  void visit(ObjectVisitor &visitor) {
    visitor.visit(*this);
  }

  void serialise(cap::Object::Builder builder) {
    Object::serialise(builder);
    auto c = builder.initController();
    c.setRight(right);
    c.setClicked(clicked);
    c.setPressed(pressed);
  }

  void deserialise(cap::Object::Reader reader) {
    Object::deserialise(reader);
    
    auto c = reader.getController();
    right = c.getRight();
    clicked = c.getClicked();
    pressed = c.getPressed();
  }

};

struct PrintVisitor : public ObjectVisitor {
  void visit(Canvas &canvas) {
    std::cout << "a canvas" << std::endl;
  }

  void visit(Controller &controller) {
    std::cout << "a controller" << std::endl;
  }

  void visit(Point &point) {
    std::cout << "a point" << std::endl;
  }

  void visit(HMD &hmd) {
    std::cout << "an HMD" << std::endl;
  }
};

struct Trigger {
  bool changed = true;
  //int nameid = -1;
  int function_nameid = -1;
  
  virtual bool check(Scene &scene) { return false; }

  virtual Trigger *copy() {
    return new Trigger(*this);
  }

  virtual void serialise(cap::Trigger::Builder builder) {
    //builder.setNameId(nameid);
    builder.setFunctionNameId(function_nameid);
  }
  
  virtual void deserialise(cap::Trigger::Reader reader) {
    //nameid = reader.getNameId();
    function_nameid = reader.getFunctionNameId();
  }
};

Trigger *read_trigger(cap::Trigger::Reader reader);
  
struct Snap {
  //state
  // all objects, with their states
  // world state vars
  // reward

  uint time = 0;
  float reward = 0;
  std::vector<uint> object_ids;
  std::vector<uint> trigger_ids;
  std::vector<uint> variable_ids;

  void serialise(cap::Snap::Builder builder) {
    builder.setTimestamp(time);
    builder.setReward(reward);

    {
      auto l = builder.initObjectIds(object_ids.size());
      for (size_t i(0); i < object_ids.size(); ++i)
        l.set(i, object_ids[i]);
    }
    {
      auto l = builder.initTriggerIds(trigger_ids.size());
      for (size_t i(0); i < trigger_ids.size(); ++i)
        l.set(i, trigger_ids[i]);
    }
    {
      auto l = builder.initVariableIds(variable_ids.size());
      for (size_t i(0); i < variable_ids.size(); ++i)
        l.set(i, variable_ids[i]);
    }    
  }
  
  void deserialise(cap::Snap::Reader reader) {
    time = reader.getTimestamp();
    reward = reader.getReward();
    object_ids.reserve(reader.getObjectIds().size());
    trigger_ids.reserve(reader.getTriggerIds().size());
    variable_ids.reserve(reader.getVariableIds().size());

    for (auto n : reader.getObjectIds())
      object_ids.push_back(n);
    
    for (auto n : reader.getTriggerIds())
      trigger_ids.push_back(n);
    
    for (auto n : reader.getVariableIds())
      variable_ids.push_back(n);    
  }

};

struct Recording {
  std::vector<Snap*> snaps;
  std::vector<Object*> objects;
  std::vector<Variable*> variables;
  std::vector<Trigger*> triggers;
  
  std::map<void*, int> index_map; //temporary data

  void deserialise(Bytes &b, Scene *scene);
  void load(std::string filename, Scene *scene);

  void serialise(Bytes *b, Scene &scene);
  void save(std::string filename, Scene &scene);

  int add_object(Object *o);
  int add_variable(Variable *v);
  int add_trigger(Trigger *t);

  int size() { return snaps.size(); }
  void load_scene(int i, Scene *scene);

  void release();
  
  //void update();
};

//struct Scene;
struct Variable {
  bool changed = true;
  int nameid = -1;
  float val = 0;

  virtual ~Variable() {}
  
  virtual void update(Scene &scene) { }
  virtual float value() {return val;}
  virtual void set_value(float val_) {
    //std::cout << "setval " << val_ << " " << val << std::endl;
    val = val_; changed = true;
  }
  virtual Variable *copy() {return new Variable(*this);}

  virtual void serialise(cap::Variable::Builder builder) {
    builder.setNameId(nameid);
    //builder.setVal(val);
  }
  
  virtual void deserialise(cap::Variable::Reader reader) {
    //std::cout << "DESERIALISE" << std::endl;
    nameid = reader.getNameId();
  }
};

Variable *read_variable(cap::Variable::Reader reader);

struct FreeVariable : public Variable {
  ~FreeVariable() {}
  
  void set_value(float val_) override {
    //std::cout << "free setval " << val_ << " " << val << std::endl;
    if (val == val_) return;
    //std::cout << "setting" << std::endl;
    val = val_;
    changed = true;
  }
  
  void serialise(cap::Variable::Builder builder) override {
    Variable::serialise(builder);
    builder.setFree(val);
    //std::cout << "free ser: " << val << std::endl;
  }

  void deserialise(cap::Variable::Reader reader) override {
    Variable::deserialise(reader);
    val = reader.getFree();
  }

  Variable *copy() override {return new FreeVariable(*this);}
};

//variable that sets itself back to 0 after storing
struct MarkVariable : public Variable {
  ~MarkVariable() {}
  
  void set_value(float val_) override {
    //std::cout << "mark setval " << val_ << " " << val << std::endl;
    val = val_;
    changed = true;
  }
  
  void update(Scene &scene) override {
    if (val) {
      val = 0;
      changed = true;
    }
  }
  
  void serialise(cap::Variable::Builder builder) override {
    Variable::serialise(builder);
    builder.setMark(val);
    //std::cout << "mark ser: " << val << std::endl;
  }

  void deserialise(cap::Variable::Reader reader) override {
    Variable::deserialise(reader);
    val = reader.getMark();
  }

  Variable *copy() override {return new MarkVariable(*this);}
};

struct Scene {
  uint time = 0;
  float reward = 0;
  bool record = false;
  bool stop = false;
  
  std::map<std::string, Object*> objects;
  std::map<std::string, Variable*> variables;
  std::vector<Trigger*> triggers;

  std::map<std::string, void_function> function_map;
  std::map<std::string, int> name_map;
  std::vector<std::string> names;
  
  int operator()(std::string name) {
    return register_name(name);
  }
  
  Object &find(int nameid) {
    return *objects[names[nameid]];
  }

  template <typename T>
  T &find(int name_id) {
    return *reinterpret_cast<T*>(objects[names[name_id]]);
  }

  template <typename T>
  T &find(std::string name) {
    if (!objects.count(name))
      throw StringException("no such object");
    return *reinterpret_cast<T*>(objects[name]);
  }
  
  Object &find(std::string name) {
    if (!objects.count(name))
      throw StringException("no such object");
    return *objects[name];
  }

  template <typename T>
  T &variable(std::string name) {
    if (!variables.count(name))
      throw StringException("no such variable");
    return *dynamic_cast<T*>(variables[name]);
  }

  Variable &variable(std::string name) {
    if (!variables.count(name))
      throw StringException("no such variable");
    return *variables[name];
  }

  void clear_scene() {
    clear_objects();
    clear_triggers();
  }
  
  void clear() {
    clear_objects(false);
    clear_triggers();
    clear_variables();
    
  }

  
  void set_tracked(bool tracked) {
   for (auto kv : objects) {
      //skip hmd and controller
     if (dynamic_cast<HMD*>(kv.second) != NULL)
       dynamic_cast<HMD*>(kv.second)->tracked = tracked;
   
     if (dynamic_cast<Controller*>(kv.second) != NULL)
       dynamic_cast<Controller*>(kv.second)->tracked = tracked;
   }
  }
  
  void clear_objects(bool filter_hmd_controller = true) {
    std::vector<std::string> to_delete;
    for (auto kv : objects) {
      //skip hmd and controller
      if (filter_hmd_controller) {
        if (dynamic_cast<HMD*>(kv.second) != NULL) continue;
        if (dynamic_cast<Controller*>(kv.second) != NULL) continue;
      }

      //delete kv.second;
      to_delete.push_back(kv.first);
    }

    for (auto d : to_delete)
      objects.erase(d);
  }

  void clear_triggers() {
    //for (auto t : triggers)
    //delete t;
    triggers.clear();
  }

  void clear_variables() {
    ////not deleting at the moment because objects might be owned by recording
    
    //for (auto kv : variables) 
    //delete kv.second;
    variables.clear();
  }

 
  
  void step();
  void snap(Recording *rec);

  void add_object(std::string name, Object *o) {
    int nameid = register_name(name);
    o->nameid = nameid;
    //if (objects.count(name))
    // delete objects[name];
    objects[name] = o;
  }

  void add_hmd() {
    add_object("hmd", new HMD());
  }
  
  void add_canvas(std::string name) {
    add_object(name, new Canvas());
  }
  
  void add_point(std::string name) {
    add_object(name, new Point());
  }

  void add_box(std::string name) {
    add_object(name, new Box());
  }
  
  
  void add_variable(std::string name, Variable *v) {
    int nameid = register_name(name);
    std::cout << "NAMEID " << nameid << std::endl;
    v->nameid = nameid;
    //if (variables.count(name))
    // delete variables[name];
    variables[name] = v;
  }


  
  void set_pos(std::string name, Pos pos) {
    Object &o = find(name);
    o.set_pos(pos);
  }

  Pos get_pos(std::string name) {
    Object &o = find(name);
    return o.p;
  }

  void set_texture(std::string name, std::string tex) {
    reinterpret_cast<Canvas&>(find(name)).set_texture(tex);
  }

  int register_name(std::string name) {
    if (name_map.count(name)) return name_map[name];
    int id = names.size();
    names.push_back(name);
    return name_map[name] = id;
  }
  
  void set_reward(float r_) {
    reward = r_;
  }

  void register_function(std::string name, void_function func) {
    function_map[name] = func;
  }
  
  void add_trigger(Trigger *t, std::string funcname);

  float dist(Pos p1, Pos p2) {
    float d(0);
    //Object &o = find(name);

    for (int i(0); i < 3; ++i)
      d += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    return sqrt(d);
  }

  void visit(ObjectVisitor &visitor) {
    visitor.i = 0;
    for (auto &kv : objects) {
      //std::cout << "visiting object " << visitor.i << std::endl;
      kv.second->visit(visitor);
      ++visitor.i;
    }
  }
  
  void start_recording() {
    //create unique id
    //grab current time
    record = true;
  }

  void end_recording() {
    //finalise, save last reward

    record = false;
  }

  void update_objects() {
    for (auto &kv : objects)
      kv.second->update();
  }

  void update_variables() {
    for (auto &kv : variables)
      kv.second->update(*this);
  }
  
  void output_json(std::ostream &out);
};

struct DistanceVariable : public Variable {
  int oid1 = -1, oid2 = -1;

  DistanceVariable() {}
  
 DistanceVariable(int oid1_, int oid2_) :
    oid1(oid1_), oid2(oid2_)
  {}

    ~DistanceVariable() {}
    
  void update(Scene &scene) {
    val = scene.dist(scene.find(oid1).p, scene.find(oid2).p);
  }

  void serialise(cap::Variable::Builder builder) {
    Variable::serialise(builder);
    auto b = builder.initDistance();
    b.setNameId1(oid1);
    b.setNameId2(oid2);
  }

  void deserialise(cap::Variable::Reader reader) {
    Variable::deserialise(reader);
    auto d = reader.getDistance();
    oid1 = d.getNameId1();
    oid2 = d.getNameId2();
  }

  Variable *copy() {return new DistanceVariable(*this);}
};

struct InBoxTrigger : public Trigger {
  bool changed = true;
  int target_id = -1, box_id = -1;

  InBoxTrigger(){}
 InBoxTrigger(int bid, int tid) : target_id(tid), box_id(bid) {}

  virtual void serialise(cap::Trigger::Builder builder) {
    Trigger::serialise(builder);
    auto l = builder.initInBox();
    l.setNameId1(target_id);
    l.setNameId2(box_id);
  }

  virtual void deserialise(cap::Trigger::Reader reader) {
    Trigger::deserialise(reader);
    auto l = reader.getInBox();
    target_id = l.getNameId1();
    box_id = l.getNameId2();
  }

  Trigger *copy() { return new InBoxTrigger(*this); }
  
  bool check(Scene &scene);  
};

struct NextTrigger : public Trigger {

  void serialise(cap::Trigger::Builder builder) {
    Trigger::serialise(builder);
    builder.setNext();
  }

  Trigger *copy() { return new NextTrigger(*this); }
  
  bool check(Scene &scene) { return true; }
};
  
struct LimitTrigger : public Trigger {
  int varnameid = -1; //var name
  float limit = 0;
  
  LimitTrigger(){}
  
 LimitTrigger(int varid, float limit_) : varnameid(varid), limit(limit_) {}

  void serialise(cap::Trigger::Builder builder) {
    Trigger::serialise(builder);
    auto l = builder.initLimit();
    l.setNameId(varnameid);
    l.setLimit(limit);    
  }

  void deserialise(cap::Trigger::Reader reader) {
    Trigger::deserialise(reader);
    auto l = reader.getLimit();
    varnameid = l.getNameId();
    limit = l.getLimit();
  }

  bool check(Scene &scene) {
    return scene.variables[scene.names[varnameid]]->val > limit;
  }

  Trigger *copy() { return new LimitTrigger(*this); }
};

struct ClickTrigger : public Trigger {
  int oid = -1;

  ClickTrigger(){}
 ClickTrigger(uint oid_) : oid(oid_) {}
  
  bool check(Scene &scene) {
     return scene.find<Controller>(oid).clicked;
  }

  void serialise(cap::Trigger::Builder builder) {
    Trigger::serialise(builder);
    builder.setClick();
  }

  void deserialise(cap::Trigger::Reader reader) {
    Trigger::deserialise(reader);
    //reader.getClick();
  }

  Trigger *copy() { return new ClickTrigger(*this); }
};

struct Action;
struct Pose {
  Pos base;
  fquat baseq;
  
  fquat armq;
  float arm_length = 0;
  bool pressed = false;
  
  Pose() {}
  Pose(Scene &scene);

  void from_scene(Scene &scene);

  void apply(Action &action);
  void apply_to_scene(Scene &scene);
  
  std::vector<float> get_vec();
  void from_vec(std::vector<float> v);
  std::vector<float> to_obs_vector();
};


inline std::ostream &operator<<(std::ostream &out, Pos &p) {
  return out << "[" << p[0] << " " << p[1] << " " << p[2] << "]"; 
}  

inline std::ostream &operator<<(std::ostream &out, fquat &q) {
  return out << "[" << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "]"; 
}  

inline std::ostream &operator<<(std::ostream &out, Pose &p) {
  return out << "P{" << p.base << " " << p.baseq << " " << p.armq << " " << p.arm_length << " " << p.pressed << "}";
}

struct Action {
  Pos dbase; //in baseq orientation
  fquat dbaseq;

  fquat armq; //is set directly
  float arm_length; //is set directly
  bool pressed; //is set directly
  
  //std::vector<float> a;
  
  Action(Pose &last, Pose &now);
  Action(std::vector<float> &a_) {
    from_vector(a_);
  }

  void from_vector(std::vector<float> &a);
  std::vector<float> to_vector();
};

inline std::ostream &operator<<(std::ostream &out, Action &a) {
  return out << "A{" << a.dbase << " " << a.dbaseq << " " << a.armq << " " << a.arm_length << " " << a.pressed << "}";
}

#endif
