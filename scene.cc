#include "scene.h"
#include "vrsystem.h"
#include "global.h"

using namespace std;

void Controller::update() {
  clicked = false;
  if (tracked) {
    auto &vr = Global::vr();
    if (right) {
      from_mat4(vr.right_controller.t);
      if (vr.right_controller.pressed && !pressed)
        clicked = true;
      pressed = vr.right_controller.pressed;
    } else {
      from_mat4(vr.left_controller.t);
      if (vr.left_controller.pressed && !pressed)
        clicked = true;
      pressed = vr.left_controller.pressed;
    }
    changed = true;
  }
}

void HMD::deserialise(cap::Object::Reader reader) {
  Object::deserialise(reader);

  //TODO remove the invert, not needed for new recordings
  auto m = Matrix4(to_mat4());
  if (Global::inst().INVERT_CORRECTION)
    m.invert();
  from_mat4(m);
}

void HMD::update() {
  if (tracked) {
    auto &vr = Global::vr();
    
    from_mat4(vr.hmd_pose);
    changed = true;
    //vr.hmd_pose = glm_to_mat4(to_mat4()); ///TODO, we need this for replaying

  }
}

void Scene::add_trigger(Trigger *t, std::string funcname) {
  int id = register_name(funcname);
  t->function_nameid = id;
  triggers.push_back(t);
  
}



void Scene::step() {
  ++time;
  
  update_objects();
  update_variables();

  auto triggers_cpy = triggers; //triggers might adjust itself
  for (auto &t : triggers_cpy) {
    if (t->check(*this)) {
      auto nameid = t->function_nameid;
      auto name = names[t->function_nameid];

      if (!function_map.count(name)) {
        cout << "trigger func " << name << " does not exist" << endl;
        throw StringException("func callback does not exist");
      }
      //cout << "trigger: " << name << endl;
      function_map[name]();
    }  
  }
}

void Scene::snap(Recording *rec) {
  Snap *snap_ptr = new Snap();
  Snap &snap(*snap_ptr);
  snap.time = time;
  snap.reward = reward;
    
  for (auto &kv : objects) {
    auto &object(*kv.second);
    if (object.changed) {//store a copy
      auto obj_cpy = object.copy();
      auto id = rec->add_object(obj_cpy);
      snap.object_ids.push_back(id);
      rec->index_map[(void*)&object] = id;
      object.changed = false;
    } else //refer to stored copy
      snap.object_ids.push_back(rec->index_map[(void*)&object]);
  }

  for (auto &kv : variables) {
    auto &variable(*kv.second);
    //cout << names[variable.nameid] << "changed?" << endl;
    if (variable.changed) {
      //cout << names[variable.nameid] << "changed" << endl;
      auto var_cpy = variable.copy();
      auto id = rec->add_variable(var_cpy);
      snap.variable_ids.push_back(id);
      rec->index_map[(void*)&variable] = id;
      variable.changed = false;
    } else
      snap.variable_ids.push_back(rec->index_map[(void*)&variable]);
  }


  for (auto &trigger_ptr : triggers) {
    Trigger &trigger(*trigger_ptr);
    if (trigger.changed) {
      auto var_cpy = trigger.copy();
      auto id = rec->add_trigger(var_cpy);
      snap.trigger_ids.push_back(id);
      rec->index_map[(void*)&trigger] = id;
      trigger.changed = false;
    } else
      snap.trigger_ids.push_back(rec->index_map[(void*)&trigger]);
  }

  rec->snaps.push_back(snap_ptr);
  
  //store snap in recording
}

void Scene::output_json(std::ostream &out) {
  JSONVisitor visitor(*this, out);

  visit(visitor);

  out << "," << endl;
  bool first(true);
  for (auto &var : variables) {
    if (var.second->changed) {
      if (!first)
        out << "," << endl;
      first = false;
      out << "\"" << var.first << "\" : " << var.second->value();
    }
  }
  out << endl;
}

bool InBoxTrigger::check(Scene &scene) {
  auto &box = scene.find<Box>(box_id);
  
  auto &target = scene.find(target_id);
  //todo: account for rotation, now we assume unrotated boxes
  bool in = target.p[0] > box.p[0] - box.width / 2 &&
    target.p[0] < box.p[0] + box.width / 2 &&
                  target.p[1] > box.p[1] - box.height / 2 &&
    target.p[1] < box.p[1] + box.height / 2 &&
                  target.p[2] > box.p[2] - box.depth / 2 &&
    target.p[2] < box.p[2] + box.depth / 2;
  
  return in;
}

Object *read_object(cap::Object::Reader reader) {
  switch (reader.which()) {
  case cap::Object::HMD: {
    auto o = new HMD();
    o->deserialise(reader);
    return o;
  }
  case cap::Object::CONTROLLER: {
    auto o = new Controller();
    o->deserialise(reader);
    return o;
  }
  case cap::Object::POINT: {
    auto o = new Point();
    o->deserialise(reader);
    return o;
  }
  case cap::Object::CANVAS: {
    auto o = new Canvas();
    o->deserialise(reader);
    return o;
  }
  case cap::Object::BOX: {
    auto o = new Box();
    o->deserialise(reader);
    return o;
  }
  default:
    throw StringException("unknown variable");
  }
}

Variable *read_variable(cap::Variable::Reader reader) {
  switch (reader.which()) {
  case cap::Variable::DISTANCE: {
    auto v = new DistanceVariable();
    v->deserialise(reader);
    return v;
  }
  case cap::Variable::FREE: {
    auto v = new FreeVariable();
    v->deserialise(reader);
    return v;
  }
  case cap::Variable::MARK: {
    auto v = new MarkVariable();
    v->deserialise(reader);
    return v;
  }
  default:
    throw StringException("unknown variable");
  }
}

Trigger *read_trigger(cap::Trigger::Reader reader) {
  switch (reader.which()) {
  case cap::Trigger::LIMIT: {
    auto t = new LimitTrigger();
    t->deserialise(reader);
    return t;
  }    
  case cap::Trigger::CLICK: {
    auto t = new ClickTrigger();
    t->deserialise(reader);
    return t;
  }
  case cap::Trigger::IN_BOX: {
    auto t = new InBoxTrigger();
    t->deserialise(reader);
    return t;
  }
  case cap::Trigger::NEXT: {
    auto t = new NextTrigger();
    t->deserialise(reader);
    return t;
  }
  default:
    throw StringException("unknown variable");
  }
}

Pose::Pose(Scene &scene) {
  from_scene(scene);
}

void Pose::from_scene(Scene &scene) {
  base = scene.find<HMD>("hmd").p;
  baseq = scene.find<HMD>("hmd").quat;

  //cout << baseq << endl;
  //throw "";
  auto c_pos = scene.find<Controller>("controller").p;
  auto v = c_pos - base;
  arm_length = l2Norm(v);

  v /= arm_length;
  cout << "v: " << v << endl;
  auto rot = glm::rotation(glm::vec3(0, 0, -1), v);
  auto v2 = glm::rotate(rot, glm::vec3(0, 0, -1));
  cout << "vn:" << v2 << endl;

  armq = glm::inverse(baseq) * rot;
  auto bla = baseq * armq;
  auto v3 = glm::rotate(bla, glm::vec3(0, 0, -1));
  auto v4 = glm::rotate(armq, glm::vec3(0, 0, -1));
  cout << "vn:" << v3 << " " << v4 << " " << endl << baseq << endl << armq << endl << bla << endl;
  
  pressed = scene.find<Controller>("controller").pressed;

}

void Pose::apply_to_scene(Scene &scene) {
  scene.find<HMD>("hmd").p = base;
  scene.find<HMD>("hmd").quat = baseq;

  cout << "quat :" << scene.find<Controller>("controller").quat << endl;
  auto blaquat = baseq * armq;
  scene.find<Controller>("controller").quat = baseq * armq;
  cout << "blaqt:" << blaquat << endl;
  cout << "quatn:" << scene.find<Controller>("controller").quat << endl;
  auto addvec = (glm::rotate(scene.find<Controller>("controller").quat, glm::vec3(0, 0, -1)) * arm_length);
  cout << "base: " << base << " " << addvec << endl;
  scene.find<Controller>("controller").p = base + glm::rotate(scene.find<Controller>("controller").quat, glm::vec3(0, 0, -1)) * arm_length;
  scene.find<Controller>("controller").pressed = pressed;
}

vector<float> Pose::get_vec() {
  vector<float> v(3 + 4 + 4 + 1 + 1);
  v[0] = base[0];
  v[1] = base[1];
  v[2] = base[2];
  
  v[3] = baseq[0];
  v[4] = baseq[1];
  v[5] = baseq[2];
  v[6] = baseq[3];

  v[7] = armq[0];
  v[8] = armq[1];
  v[9] = armq[2];
  v[10] = armq[3];

  v[11] = arm_length;
  v[12] = pressed;
  return v;
}

void Pose::from_vec(std::vector<float> v) {
  base[0] = v[0];
  base[1] = v[1];
  base[2] = v[2];
  
  baseq[0] = v[3];
  baseq[1] = v[4];
  baseq[2] = v[5];
  baseq[3] = v[6];

  armq[0] = v[7];
  armq[1] = v[8];
  armq[2] = v[9];
  armq[3] = v[10];

  baseq = glm::normalize(baseq);
  armq = glm::normalize(armq);
  
  arm_length = v[11];
  if (arm_length < 0)
    arm_length = 0;
  pressed = v[12] > .5;
}

std::vector<float> Pose::to_obs_vector() {
  vector<float> v(6);
  v[0] = armq[0];
  v[1] = armq[1];
  v[2] = armq[2];
  v[3] = armq[3];

  v[4] = arm_length;
  v[5] = 0;//pressed; //for learning no cheating or it will copy it
  return v;
}


void Pose::apply(Action &act) {
  base += glm::rotate(baseq, act.dbase);
  baseq = act.dbaseq * baseq;
  
  armq = act.armq;
  arm_length = act.arm_length;
  pressed = act.pressed;
}

Action::Action(Pose &last, Pose &now) {
  dbase = glm::rotate(glm::inverse(last.baseq), now.base - last.base);
  dbaseq = now.baseq * glm::inverse(last.baseq);

  armq = now.armq;
  arm_length = now.arm_length;
  pressed = now.pressed;
    
}

void Action::from_vector(std::vector<float> &v) {
  dbase[0] = v[0];
  dbase[1] = v[1];
  dbase[2] = v[2];
  
  dbaseq[0] = v[3];
  dbaseq[1] = v[4];
  dbaseq[2] = v[5];
  dbaseq[3] = v[6];

  armq[0] = v[7];
  armq[1] = v[8];
  armq[2] = v[9];
  armq[3] = v[10];

  dbaseq = glm::normalize(dbaseq);
  armq = glm::normalize(armq);
  
  arm_length = v[11];
  if (arm_length < 0)
    arm_length = 0;
  pressed = v[12] > .5;
}

std::vector<float> Action::to_vector() {
  std::vector<float> v(3 + 4 + 4 + 1 + 1);
  v[0] = dbase[0];
  v[1] = dbase[1];
  v[2] = dbase[2];
 
  v[3] = dbaseq[0];
  v[4] = dbaseq[1];
  v[5] = dbaseq[2];
  v[6] = dbaseq[3];

  v[7] = armq[0];
  v[8] = armq[1];
  v[9] = armq[2];
  v[10] = armq[3];

  v[11] = arm_length;
  v[12] = pressed;
  return v;
}

void JSONVisitor::visit(Canvas &canvas) {
  throw StringException("not implemented");
}

void JSONVisitor::visit(Controller &controller) {
  if (!object_json_output(controller))
    return;
  out << "," << endl;
  out << "\"right\": " << controller.right << "," << endl;
  out << "\"pressed\": " << controller.pressed << "," << endl;
  out << "\"clicked\": " << controller.clicked << endl;
  out << "}" << endl;
}

void JSONVisitor::visit(Point &point) {
  if (!object_json_output(point))
    return;
  out << "}" << endl;
}


void JSONVisitor::visit(HMD &hmd) {
  if (!object_json_output(hmd))
    return;
  out << "}" << endl;
}

void JSONVisitor::visit(Box &box) {
  if (!object_json_output(box))
    return;
  out << "," << endl;
  out << "\"width\": " << box.width << "," << endl;
  out << "\"height\": " << box.height << "," << endl;
  out << "\"depth\": " << box.depth << "," << endl;
  out << "\"tex\": \"" << box.tex_name << "\"" << endl;
  out << "}" << endl;
}

//leaves bracket open! needs '}'
bool JSONVisitor::object_json_output(Object &o) {
  if (!o.changed)
    return false;
  string name = scene.names[o.nameid];

  auto p = o.p;
  auto q = o.quat;
  if (!first)
    out << "," << endl;
  first = false;
  out << "\"" << name << "\": {" << endl;
  out << "\"pos\": [" << p.x << "," << p.y << "," << p.z << "]," << endl;
  out << "\"quat\": [" << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "]";
  
  return true;
}
