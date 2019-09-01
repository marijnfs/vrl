#include "scene.h"
#include "bytes.h"
#include "serialise.h"
#include "gzstream.h"
#include "util.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <capnp/serialize-packed.h>

#include <algorithm>
#include <iterator>

using namespace std;

void Recording::load_scene(int i, Scene *scene_ptr) {
  Scene &scene(*scene_ptr);
  
  scene.clear();
  auto &snap = *snaps[i];

  //todo
  scene.time = snap.time;
  scene.reward = snap.reward;
  
  for (auto o : snap.object_ids) {
    string name = scene.names[objects[o]->nameid];
    scene.objects[name] = objects[o];
  }
  
  for (auto v : snap.variable_ids) {
    string name = scene.names[variables[v]->nameid];
    scene.variables[name] = variables[v];
  }

  for (auto t : snap.trigger_ids)
    scene.triggers.push_back(triggers[t]);
  
}

void Recording::serialise(Bytes *bytes, Scene &scene) {
  ::capnp::MallocMessageBuilder cap_message;
  auto builder = cap_message.initRoot<cap::Recording>();
  
  {
    int i(0);
    auto name_builder = builder.initNames(scene.names.size());
    for (auto n : scene.names)
      name_builder.set(i++, n);
  }
  //for (int i(0); i < scene.objects.size(); ++i)
  {
    int i(0);
    auto object_builder = builder.initObjects(objects.size());
    for (auto o : objects)
      o->serialise(object_builder[i++]);
  }
  
  {
    int i(0);
    auto var_builder = builder.initVariables(variables.size());
    for (auto v : variables)
      v->serialise(var_builder[i++]);
  }

  {
    int i(0);
    auto trig_builder = builder.initTriggers(triggers.size());
    for (auto &t : triggers)
      t->serialise(trig_builder[i++]);
  }

  {
    int i(0);
    auto snap_builder = builder.initSnaps(snaps.size());
    for (auto &s : snaps)
      s->serialise(snap_builder[i++]);
  }

  auto cap_data = messageToFlatArray(cap_message);
  auto cap_bytes = cap_data.asBytes();
  bytes->resize(cap_bytes.size());
  memcpy(&(*bytes)[0], &cap_bytes[0], cap_bytes.size());
  //copy(cap_bytes.begin(), cap_bytes.end(), &bytes[0]);
}

int Recording::add_object(Object *o) {
  int idx = objects.size();
  objects.push_back(o);
  index_map[(void*)o] = idx;
  return idx;
}

int Recording::add_variable(Variable *v) {
  int idx = variables.size();
  variables.push_back(v);
  index_map[(void*)v] = idx;
  return idx;
}

int Recording::add_trigger(Trigger *t) {
  int idx = triggers.size();
  triggers.push_back(t);
  index_map[(void*)t] = idx;
  return idx;
}

void Recording::deserialise(Bytes &bytes, Scene *scene) {
  ::capnp::FlatArrayMessageReader reader(bytes.kjwp());
  variables.clear();
  objects.clear();
  triggers.clear();
  snaps.clear();
  index_map.clear();
  
  auto rec = reader.getRoot<cap::Recording>();

  auto rec_names = rec.getNames();
  auto rec_objects = rec.getObjects();
  auto rec_variables = rec.getVariables();
  auto rec_triggers = rec.getTriggers();
  auto rec_snaps = rec.getSnaps();

  for (auto n : rec_names)
    scene->register_name(n);

  for (auto v : rec_variables) {
    add_variable(read_variable(v));
  }
  
  for (auto t : rec_triggers)
    add_trigger(read_trigger(t));
  
  for (auto ob : rec_objects)
    add_object(read_object(ob));

  for (auto rsnap : rec_snaps) {
    auto snap = new Snap();
    snap->deserialise(rsnap);
    snaps.push_back(snap);
  }

}

void Recording::load(std::string filename, Scene *scene) {
  igzstream ifs(filename.c_str());
  if (!ifs)
    throw StringException("couldn't open file");
  Bytes b;
  //ifs.seekg(0, std::ios::end);
  //int end = ifs.tellg();
  //cout << "tellg: " << end << endl;
  //if (end < 0)
  // throw StringException("couldn't read file");
    
  //b.reserve(end);
  //ifs.seekg(0, std::ios::beg);
      
  copy(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>(), back_insert_iterator<Bytes>(b));
  deserialise(b, scene);

  /*for (auto v : variables) {
    cout << scene->names[v->nameid] << " ";
  }
  cout << endl;
  
  for (auto snap : snaps) {
    cout << snap->time << ": ";
    for (auto id : snap->variable_ids) {
      cout << id << " ";
    }
    cout << endl;
  }
  throw "";*/
}

void Recording::save(std::string filename, Scene &scene) {
  cout << "SAVING TO " << filename << endl;
  Bytes b;
  serialise(&b, scene);

  ogzstream of(filename.c_str());
  of.write((char*)&b[0], b.size());
  cout << "Done SAVING" << endl;
}

void Recording::release() {
  for (auto v : variables)
    delete v;
  for (auto t : triggers)
    delete t;
  for (auto o : objects)
    delete o;
  for (auto s : snaps)
    delete s;
}
